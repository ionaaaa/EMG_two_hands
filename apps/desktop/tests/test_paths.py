import json

from emg_live_marker.paths import ProjectPaths, resolve_project_paths


def test_defaults_use_repository_data_locations(tmp_path):
    paths = resolve_project_paths(project_root=tmp_path, environ={})

    assert paths == ProjectPaths(
        project_root=tmp_path,
        dataset_root=tmp_path / "data" / "datasets",
        recordings_root=tmp_path / "data" / "recordings",
        artifacts_root=tmp_path / "apps" / "desktop",
        config_path=None,
    )
    assert paths.models_root == tmp_path / "apps" / "desktop" / "models"
    assert paths.reports_root == tmp_path / "apps" / "desktop" / "reports"


def test_cli_values_override_config_environment_and_defaults(tmp_path):
    config = tmp_path / "paths.json"
    config.write_text(
        json.dumps(
            {
                "dataset_root": "config/dataset",
                "recordings_root": "config/recordings",
                "artifacts_root": "config/artifacts",
            }
        ),
        encoding="utf-8",
    )
    paths = resolve_project_paths(
        project_root=tmp_path,
        paths_config=config,
        dataset_root="cli/dataset",
        environ={
            "EMG_DATASET_ROOT": "env/dataset",
            "EMG_RECORDINGS_ROOT": "env/recordings",
            "EMG_ARTIFACTS_ROOT": "env/artifacts",
        },
    )

    assert paths.dataset_root == tmp_path / "cli" / "dataset"
    assert paths.recordings_root == tmp_path / "config" / "recordings"
    assert paths.artifacts_root == tmp_path / "config" / "artifacts"


def test_environment_values_override_legacy_defaults(tmp_path):
    paths = resolve_project_paths(
        project_root=tmp_path,
        environ={
            "EMG_DATASET_ROOT": "env/dataset",
            "EMG_RECORDINGS_ROOT": "env/recordings",
            "EMG_ARTIFACTS_ROOT": "env/artifacts",
        },
    )

    assert paths.dataset_root == tmp_path / "env" / "dataset"
    assert paths.recordings_root == tmp_path / "env" / "recordings"
    assert paths.artifacts_root == tmp_path / "env" / "artifacts"


def test_relative_values_are_resolved_from_project_root_not_cwd(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    paths = resolve_project_paths(project_root=tmp_path, dataset_root="data/datasets")

    assert paths.dataset_root == tmp_path / "data" / "datasets"
