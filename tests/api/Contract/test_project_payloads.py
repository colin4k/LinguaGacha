from api.Contract.ProjectPayloads import ProjectPreviewPayload
from api.Contract.ProjectPayloads import ProjectSnapshotPayload


def test_project_snapshot_payload_keeps_loaded_state() -> None:
    payload = ProjectSnapshotPayload(path="demo/project.lg", loaded=True).to_dict()

    assert payload == {
        "path": "demo/project.lg",
        "loaded": True,
    }


def test_project_preview_payload_normalizes_optional_summary_fields() -> None:
    payload = ProjectPreviewPayload.from_dict(
        {
            "path": "demo/project.lg",
            "name": "Demo",
            "source_language": "JA",
            "target_language": "ZH",
            "file_count": 4,
            "translation_stats": {
                "total_items": 12,
                "completed_count": 3,
                "failed_count": 2,
                "pending_count": 6,
                "skipped_count": 1,
                "completion_percent": 33.33,
            },
        }
    ).to_dict()

    assert payload == {
        "path": "demo/project.lg",
        "name": "Demo",
        "source_language": "JA",
        "target_language": "ZH",
        "file_count": 4,
        "created_at": "",
        "updated_at": "",
        "translation_stats": {
            "total_items": 12,
            "completed_count": 3,
            "failed_count": 2,
            "pending_count": 6,
            "skipped_count": 1,
            "completion_percent": 33.33,
        },
    }


def test_project_preview_payload_defaults_missing_summary_to_empty_preview() -> None:
    payload = ProjectPreviewPayload.from_dict(None).to_dict()

    assert payload == {
        "path": "",
        "name": "",
        "source_language": "",
        "target_language": "",
        "file_count": 0,
        "created_at": "",
        "updated_at": "",
        "translation_stats": {
            "total_items": 0,
            "completed_count": 0,
            "failed_count": 0,
            "pending_count": 0,
            "skipped_count": 0,
            "completion_percent": 0.0,
        },
    }
