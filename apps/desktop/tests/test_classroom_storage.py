import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from emg_live_marker.paths import resolve_project_paths
from emg_live_marker.realtime.classroom_storage import (
    ClassroomStorage,
    ClassroomStorageError,
)
from emg_live_marker.realtime.game_mapping import GameMappingService
from emg_live_marker.realtime.student_competition import StudentCompetitionService
from emg_live_marker.realtime.student_personal_training import StudentPersonalTrainingService
from emg_live_marker.realtime.teacher_classroom import TeacherClassroomService


def course_config() -> dict:
    path = Path(__file__).parents[3] / "configs" / "teaching" / "yucai.json"
    return json.loads(path.read_text(encoding="utf-8"))


def mapping_config() -> dict:
    return course_config()


def storage(tmp_path: Path) -> ClassroomStorage:
    return ClassroomStorage.from_config(
        tmp_path / "data" / "classroom",
        course_config(),
        app_data_root=tmp_path / "app-data" / "EMGTwoHands",
    )


@pytest.mark.parametrize(
    "bad_id",
    ("", ".", "..", "../group", "group/one", "group\\one", "C:\\group", "/tmp/group"),
)
def test_group_id_rejects_traversal_and_absolute_paths(tmp_path, bad_id) -> None:
    with pytest.raises(ClassroomStorageError):
        storage(tmp_path).group_paths(bad_id)


def test_group_directory_profile_and_two_group_isolation(tmp_path) -> None:
    classroom = storage(tmp_path)
    first = classroom.group_paths("group_01", create=True)
    second = classroom.group_paths("group_02", create=True)
    assert first.root == (
        tmp_path / "data" / "classroom" / "yucai_2026" / "group_01"
    ).resolve()
    assert first.sessions.is_dir() and first.models.is_dir() and first.results.is_dir()
    assert second.root != first.root
    classroom.write_settings("group_01", {"sensitivity": "high"})
    classroom.write_settings("group_02", {"sensitivity": "low"})
    assert classroom.read_settings("group_01")["sensitivity"] == "high"
    assert classroom.read_settings("group_02")["sensitivity"] == "low"
    profile = json.loads(first.profile.read_text(encoding="utf-8"))
    assert set(profile) == {
        "schema_version",
        "student_id",
        "course_id",
        "classroom_id",
        "created_at",
        "updated_at",
    }
    assert profile["student_id"] == "group_01"
    assert profile["course_id"] == "yucai"
    assert profile["classroom_id"] == "yucai_2026"
    assert "name" not in profile


def test_atomic_settings_and_all_generated_paths(tmp_path, monkeypatch) -> None:
    classroom = storage(tmp_path)
    real_replace = os.replace
    calls = []

    def tracked_replace(source, destination):
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("emg_live_marker.realtime.classroom_storage.os.replace", tracked_replace)
    settings = classroom.write_settings(
        "group_01",
        {
            "mapping": {"rest": "none", "fist": "A", "open-palm": "B", "pinch": "none"},
            "sensitivity": "high",
            "control_style": "stable",
            "active_model": {"type": "personal", "run_id": "run_01"},
            "recollect_requested": ["session_01"],
        },
    )
    paths = classroom.group_paths("group_01")
    assert calls and calls[-1][1] in {paths.settings, paths.profile}
    assert not list(paths.root.rglob("*.tmp"))
    assert settings["active_model"] == {"type": "personal", "run_id": "run_01"}
    assert classroom.session_path("group_01", "session_01") == paths.sessions / "session_01"
    assert classroom.model_path("group_01", "run_01") == paths.models / "run_01"
    assert classroom.result_path("group_01", "round_01") == paths.results / "round_01.json"


def test_mapping_and_control_profile_restore_after_restart(tmp_path) -> None:
    classroom = storage(tmp_path)
    mapping = GameMappingService(mapping_config(), classroom_storage=classroom)
    mapping.swap_commands()
    mapping.set_control_preferences("high", "stable")
    assert mapping.save_current_group("group_01")[0]

    restarted_storage = storage(tmp_path)
    restarted = GameMappingService(mapping_config(), classroom_storage=restarted_storage)
    assert restarted.current_group_id == "group_01"
    assert restarted.resolved_mapping["fist"] == "B"
    assert restarted.control_preferences == {
        "sensitivity": "high",
        "control_style": "stable",
    }
    persisted = restarted_storage.read_settings("group_01")
    assert "control_profile" not in persisted
    assert persisted["sensitivity"] == "high"
    assert persisted["control_style"] == "stable"


class FakeObservation:
    def __init__(self, standard_path: Path) -> None:
        self.settings = SimpleNamespace(model_path=standard_path)
        self.model_source = "standard"
        self.personal_model_available = False

    def activate_personal_model(self, _path, _predictor=None):
        self.model_source = "personal"
        self.personal_model_available = True
        return True

    def use_standard_model(self):
        self.model_source = "standard"
        return True

    def use_personal_model(self):
        if not self.personal_model_available:
            return False
        self.model_source = "personal"
        return True


def test_missing_personal_run_safely_falls_back_to_standard(tmp_path) -> None:
    classroom = storage(tmp_path)
    classroom.write_settings(
        "group_01",
        {"active_model": {"type": "personal", "run_id": "missing_run"}},
    )
    standard = tmp_path / "standard.pt"
    standard.write_bytes(b"standard")
    observation = FakeObservation(standard)
    service = StudentPersonalTrainingService(
        tmp_path,
        tmp_path / "data" / "datasets",
        course_config(),
        observation,
        classroom_storage=classroom,
    )
    restored, message = service.restore_model_selection("group_01")
    assert restored is False
    assert "回退到标准模型" in message
    assert observation.model_source == "standard"
    assert classroom.read_settings("group_01")["active_model"] == {
        "type": "standard",
        "run_id": None,
    }


def test_existing_personal_run_restores_without_storing_absolute_path(tmp_path) -> None:
    classroom = storage(tmp_path)
    model_dir = classroom.model_path("group_01", "run_01", create=True)
    model = model_dir / "gesture_classifier.pt"
    model.write_bytes(b"personal")
    for name in (
        "gesture_labels.json",
        "normalization.json",
        "model_info.json",
        "train_report.json",
    ):
        ClassroomStorage.atomic_write_json(model_dir / name, {})
    ClassroomStorage.atomic_write_json(
        model_dir / "personal_model.valid.json",
        {"schema_version": 1, "anonymous_group_id": "group_01", "run_id": "run_01"},
    )
    classroom.write_settings(
        "group_01",
        {"active_model": {"type": "personal", "run_id": "run_01"}},
    )
    standard = tmp_path / "standard.pt"
    standard.write_bytes(b"standard")
    observation = FakeObservation(standard)
    service = StudentPersonalTrainingService(
        tmp_path,
        tmp_path / "data" / "datasets",
        course_config(),
        observation,
        classroom_storage=classroom,
        model_loader=lambda _path: object(),
    )
    restored, message = service.restore_model_selection("group_01")
    assert restored and message == "已恢复我的模型。"
    assert observation.model_source == "personal"
    settings_text = classroom.group_paths("group_01").settings.read_text(encoding="utf-8")
    assert str(model.resolve()) not in settings_text
    assert json.loads(settings_text)["active_model"] == {
        "type": "personal",
        "run_id": "run_01",
    }


def complete_session(path: Path, group_id: str) -> None:
    path.mkdir(parents=True)
    metadata = {
        "anonymous_id": group_id,
        "session_id": path.name,
        "selected_hand": "left",
        "collection_status": "completed",
        "valid_trial_counts": {"fist": 5, "finger_spread": 5, "thumb_index_pinch": 5},
        "trial_statuses": {},
    }
    ClassroomStorage.atomic_write_json(path / "metadata.json", metadata)
    (path / "emg.csv").write_text("sample_index\n", encoding="utf-8")
    (path / "events.csv").write_text("event\n", encoding="utf-8")


def test_training_discovers_new_and_legacy_sessions_without_modifying_legacy(tmp_path) -> None:
    classroom = storage(tmp_path)
    new_session = classroom.session_path("group_new", "session_new")
    complete_session(new_session, "group_new")
    legacy = tmp_path / "data" / "datasets" / "group_old" / "session_old"
    complete_session(legacy, "group_old")
    before = (legacy / "metadata.json").read_bytes()
    standard = tmp_path / "standard.pt"
    standard.write_bytes(b"standard")
    service = StudentPersonalTrainingService(
        tmp_path,
        tmp_path / "data" / "datasets",
        course_config(),
        FakeObservation(standard),
        classroom_storage=classroom,
    )
    sessions = service.discover_sessions()
    assert {(item.group_id, item.session_id) for item in sessions} == {
        ("group_new", "session_new"),
        ("group_old", "session_old"),
    }
    assert (legacy / "metadata.json").read_bytes() == before


class CompetitionObservation:
    model_source = "standard"
    current_sensitivity = "standard"
    current_control_style = "balanced"


def test_competition_and_teacher_scan_new_results_plus_legacy_read_only(tmp_path) -> None:
    config = course_config()
    classroom = storage(tmp_path)
    mapping = GameMappingService(config, classroom_storage=classroom)
    competition = StudentCompetitionService(
        tmp_path,
        config,
        CompetitionObservation(),
        mapping,
        classroom_storage=classroom,
    )
    assert competition.begin_competition("group_01")[0]
    status, _response = competition.save_browser_result(
        {
            "mode": "left",
            "score": 100,
            "accuracy": 0.75,
            "max_combo": 4,
            "outcome": "completed",
        },
        result_id="round-01",
    )
    assert status == 201
    assert competition.latest_result_path.parent == classroom.group_paths("group_01").results
    assert classroom.read_settings("group_01")["mapping"] == mapping.resolved_mapping

    paths = resolve_project_paths(project_root=tmp_path, environ={})
    standard_dir = paths.models_root / "standard"
    standard_dir.mkdir(parents=True)
    standard = standard_dir / "gesture_classifier.pt"
    standard.write_bytes(b"model")
    for name in ("gesture_labels.json", "normalization.json", "model_info.json"):
        (standard_dir / name).write_text("{}", encoding="utf-8")
    config["realtime_decoding"]["standard_teaching_model_path"] = str(standard)
    teacher = TeacherClassroomService(
        paths,
        config,
        classroom_storage=classroom,
        settings_path=tmp_path / "app-data" / "classroom_settings.json",
        model_loader=lambda _path: SimpleNamespace(model_type="real"),
    )
    records = teacher.scan_competition_results()
    assert len(records) == 1 and records[0]["student_id"] == "group_01"
    export = tmp_path / "export" / "competition.csv"
    assert teacher.export_competition_csv(export, student_id="group_01")[0]
    assert "group_01" in export.read_text(encoding="utf-8-sig")

    legacy = paths.dataset_root / "group_old" / "session_old"
    complete_session(legacy, "group_old")
    assert any(item.legacy for item in teacher.scan_sessions())
    legacy_metadata = (legacy / "metadata.json").read_bytes()
    assert teacher.mark_recollect(legacy)[0]
    assert (legacy / "metadata.json").read_bytes() == legacy_metadata
    assert not (legacy / "recollect_requested.json").exists()
    assert teacher.delete_session(legacy, confirmed=True)[0] is False
    assert legacy.is_dir()

    legacy_model = paths.models_root / "student_personal" / "group_old_run"
    legacy_model.mkdir(parents=True)
    (legacy_model / "gesture_classifier.pt").write_bytes(b"personal")
    ClassroomStorage.atomic_write_json(
        legacy_model / "personal_model.valid.json",
        {"anonymous_group_id": "group_old", "validated_at": "2026-09-02T00:00:00Z"},
    )
    ClassroomStorage.atomic_write_json(
        legacy_model / "train_report.json", {"val_accuracy": 0.8}
    )
    legacy_records = [item for item in teacher.scan_personal_models() if item.legacy]
    assert len(legacy_records) == 1
    assert teacher.delete_personal_model(legacy_model, confirmed=True)[0] is False
    assert legacy_model.is_dir()
