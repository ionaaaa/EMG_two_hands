import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from emg_live_marker.cli import finetune_effie_gesture as finetune
from emg_live_marker.realtime.student_personal_training import (
    REQUIRED_ARTIFACTS,
    StudentPersonalTrainingService,
)
from emg_live_marker.ui.student_pages import COURSE_ENTRIES, StudentPersonalTrainingPage


class FakeObservation(QObject):
    model_source_changed = Signal(str, str)

    def __init__(self, standard_path: Path) -> None:
        super().__init__()
        self.settings = SimpleNamespace(model_path=standard_path)
        self.model_source = "standard"
        self.personal_model_available = False
        self.personal_path = None

    def activate_personal_model(self, path, predictor=None) -> bool:
        self.personal_path = Path(path)
        self.personal_model_available = True
        self.model_source = "personal"
        self.model_source_changed.emit("personal", "我的模型")
        return True

    def use_standard_model(self) -> bool:
        self.model_source = "standard"
        self.model_source_changed.emit("standard", "标准模型")
        return True

    def use_personal_model(self) -> bool:
        if not self.personal_model_available:
            return False
        self.model_source = "personal"
        self.model_source_changed.emit("personal", "我的模型")
        return True


class FakeProcess(QObject):
    readyReadStandardOutput = Signal()
    finished = Signal(int, object)
    errorOccurred = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.program = ""
        self.arguments = []
        self.output = b""
        self.started = False
        self.terminated = False
        self.killed = False

    def setProgram(self, value):
        self.program = value

    def setArguments(self, value):
        self.arguments = list(value)

    def setWorkingDirectory(self, _value):
        pass

    def setProcessEnvironment(self, _value):
        pass

    def setProcessChannelMode(self, _value):
        pass

    def start(self):
        self.started = True

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def waitForFinished(self, _milliseconds):
        return not self.started

    def readAllStandardOutput(self):
        output, self.output = self.output, b""
        return output

    def emit_output(self, text):
        self.output += text.encode("utf-8")
        self.readyReadStandardOutput.emit()


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make_session(root: Path, *, statuses=True, completed=5) -> Path:
    session = root / "group_01" / "session_001"
    session.mkdir(parents=True)
    (session / "emg.csv").write_text("sample_index\n", encoding="utf-8")
    rows = ["trial_id,phase,gesture,sample_index,note"]
    trial_statuses = {}
    valid_counts = {}
    trial = 0
    for gesture in ("fist", "finger_spread", "thumb_index_pinch"):
        valid_counts[gesture] = completed
        for _ in range(completed):
            trial += 1
            trial_id = str(trial)
            rows.append(f"{trial_id},gesture_start,{gesture},{trial * 100},")
            rows.append(f"{trial_id},trial_end,{gesture},{trial * 100 + 50},")
            trial_statuses[trial_id] = "completed"
        trial += 1
        rows.append(f"{trial},gesture_start,{gesture},{trial * 100},")
        rows.append(f"{trial},trial_end,{gesture},{trial * 100 + 50},invalid")
        trial_statuses[str(trial)] = "invalid"
    (session / "events.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    metadata = {"valid_trial_counts": valid_counts}
    if statuses:
        metadata["trial_statuses"] = trial_statuses
    (session / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return session


def make_service(tmp_path, *, completed=5, loader=lambda _path: object()):
    standard = tmp_path / "standard" / "gesture_classifier.pt"
    standard.parent.mkdir(parents=True)
    standard.write_bytes(b"standard-read-only")
    dataset = tmp_path / "data" / "datasets"
    session = make_session(dataset, completed=completed)
    observation = FakeObservation(standard)
    processes = []

    def process_factory(parent):
        process = FakeProcess(parent)
        processes.append(process)
        return process

    config = {
        "personal_training": {
            "minimum_completed_trials_per_gesture": 5,
            "epochs": 10,
            "output_directory": str(tmp_path / "personal"),
        }
    }
    service = StudentPersonalTrainingService(
        tmp_path, dataset, config, observation, process_factory=process_factory, model_loader=loader
    )
    return service, observation, processes, session, standard


def write_artifacts(output_dir: Path, *, omit=()) -> None:
    report = {
        "val_accuracy": 0.875,
        "per_class_precision_recall_f1": {
            label: {"f1": value}
            for label, value in zip(("rest", "fist", "open-palm", "pinch"), (0.9, 0.8, 0.85, 0.7))
        },
    }
    for name in REQUIRED_ARTIFACTS:
        if name in omit:
            continue
        content = json.dumps(report) if name == "train_report.json" else "{}"
        (output_dir / name).write_text(content, encoding="utf-8")


def test_training_lesson_is_available() -> None:
    entry = next(item for item in COURSE_ENTRIES if item.identifier == "train-model")
    assert entry.available is True
    assert entry.title == "训练我的模型"


def test_discovers_anonymous_sessions_and_counts_only_completed(tmp_path) -> None:
    service, _observation, _processes, session, _standard = make_service(tmp_path)
    inspected = service.discover_sessions()[0]
    assert inspected.path == session
    assert inspected.group_id == "group_01"
    assert inspected.ready is True
    assert inspected.counts == {"fist": 5, "finger_spread": 5, "thumb_index_pinch": 5}
    assert "握拳 5 次" in inspected.message


def test_incomplete_metadata_blocks_training_with_chinese_counts(tmp_path) -> None:
    service, _observation, processes, _session, _standard = make_service(tmp_path, completed=4)
    inspected = service.discover_sessions()[0]
    assert inspected.ready is False
    assert service.start_training(inspected) is False
    assert not processes
    assert "每类至少需要 5 次" in service.snapshot.message


def test_qprocess_command_progress_duplicate_guard_and_cancel(tmp_path) -> None:
    service, _observation, processes, _session, standard = make_service(tmp_path)
    selected = service.discover_sessions()[0]
    assert service.start_training(selected) is True
    process = processes[0]
    assert process.program == os.sys.executable
    assert process.arguments[:3] == ["-u", "-m", "emg_live_marker.cli.finetune_effie_gesture"]
    assert process.arguments[process.arguments.index("--checkpoint-path") + 1] == str(standard.resolve())
    assert process.arguments[process.arguments.index("--mode") + 1] == "freeze_backbone"
    assert service.start_training(selected) is False
    process.emit_output("epoch 003 train_acc=0.7500 val_acc=0.6250\n")
    assert service.snapshot.epoch == 3
    assert service.snapshot.validation_accuracy == pytest.approx(0.625)
    assert service.cancel_training() is True
    assert process.terminated is True
    service._kill_if_running()
    assert process.killed is True
    process.finished.emit(1, None)
    assert service.snapshot.state == "cancelled"
    assert service.temporary_output_dir is None


def test_close_terminates_and_kills_running_training(tmp_path) -> None:
    service, _observation, processes, _session, _standard = make_service(tmp_path)
    assert service.start_training(service.discover_sessions()[0])
    output_dir = service.temporary_output_dir
    service.close()
    assert processes[0].terminated is True
    assert processes[0].killed is True
    assert service.running is False
    assert not output_dir.exists()


def test_success_requires_all_artifacts_loads_and_activates_personal_model(tmp_path) -> None:
    service, observation, processes, _session, standard = make_service(tmp_path)
    before = standard.read_bytes()
    assert service.start_training(service.discover_sessions()[0])
    write_artifacts(service.temporary_output_dir)
    processes[0].finished.emit(0, None)
    assert service.snapshot.state == "completed"
    assert service.snapshot.validation_accuracy == pytest.approx(0.875)
    assert service.snapshot.per_class_performance["pinch"] == pytest.approx(0.7)
    assert service.last_valid_model_dir.is_dir()
    assert (service.last_valid_model_dir / "personal_model.valid.json").is_file()
    assert observation.model_source == "personal"
    assert observation.personal_path == service.last_valid_model_dir / "gesture_classifier.pt"
    assert standard.read_bytes() == before
    assert service.use_standard_model() is True
    assert observation.model_source == "standard"
    assert service.use_personal_model() is True
    assert observation.model_source == "personal"


def test_missing_artifact_or_failed_process_keeps_previous_model(tmp_path) -> None:
    service, observation, processes, _session, standard = make_service(tmp_path)
    before = standard.read_bytes()
    assert service.start_training(service.discover_sessions()[0])
    write_artifacts(service.temporary_output_dir, omit={"normalization.json"})
    processes[0].finished.emit(0, None)
    assert service.snapshot.state == "failed"
    assert "normalization.json" in service.snapshot.message
    assert observation.model_source == "standard"
    assert standard.read_bytes() == before

    assert service.start_training(service.discover_sessions()[0])
    processes[1].emit_output("真实训练错误\n")
    processes[1].finished.emit(2, None)
    assert service.snapshot.state == "failed"
    assert "真实训练错误" in service.snapshot.message
    assert observation.model_source == "standard"


def test_training_page_has_no_technical_controls(app, tmp_path) -> None:
    service, _observation, _processes, _session, _standard = make_service(tmp_path)
    page = StudentPersonalTrainingPage(service, lambda: None)
    page.activate()
    assert page.group_combo.currentText() == "group_01"
    assert page.train_button.isEnabled()
    assert "5 次" in page.count_labels["fist"].text()
    visible_text = " ".join(button.text() for button in page.findChildren(type(page.train_button)))
    assert "学习率" not in visible_text
    assert "epoch" not in visible_text.lower()


def test_effie_collection_excludes_non_completed_trials_and_keeps_legacy(monkeypatch, tmp_path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    phases = []
    for trial_id in ("done", "invalid", "repeated", "interrupted"):
        for phase, sample in (
            ("trial_start", 0), ("gesture_start", 1000), ("gesture_end", 3000), ("trial_end", 4000)
        ):
            phases.append({"trial_id": trial_id, "phase": phase, "gesture": "fist", "sample_index": str(sample)})
    statuses = {"done": "completed", "invalid": "invalid", "repeated": "repeated", "interrupted": "interrupted"}
    (session / "metadata.json").write_text(json.dumps({"trial_statuses": statuses}), encoding="utf-8")
    monkeypatch.setattr(finetune, "discover_session_dirs", lambda _root: [session])
    monkeypatch.setattr(
        finetune,
        "load_emg",
        lambda *_args, **_kwargs: (np.arange(5000), np.arange(5000), np.zeros((5000, 8))),
    )
    monkeypatch.setattr(finetune, "read_csv_dicts", lambda _path: phases)
    monkeypatch.setattr(
        finetune,
        "slice_effie_windows",
        lambda _segment, label: [(np.zeros((400, 8), dtype=np.float32), label)],
    )
    assert {record.trial_id for record in finetune.collect_effie_records(tmp_path)} == {"done"}
    (session / "metadata.json").write_text("{}", encoding="utf-8")
    assert {record.trial_id for record in finetune.collect_effie_records(tmp_path)} == set(statuses)
