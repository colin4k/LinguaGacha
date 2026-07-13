from dataclasses import dataclass
from dataclasses import field
from typing import Any

from api.Models.Project import ProjectTranslationStats


@dataclass(frozen=True)
class ProjectSnapshotPayload:
    """统一描述工程加载快照，避免把 DataManager 的内部状态直接外泄。"""

    path: str
    loaded: bool

    def to_dict(self) -> dict[str, Any]:
        """转换为稳定 JSON 结构，供 HTTP 响应载荷使用。"""

        return {
            "path": self.path,
            "loaded": self.loaded,
        }


@dataclass(frozen=True)
class ProjectPreviewPayload:
    """统一描述打开工程页使用的工程摘要载荷。"""

    path: str = ""
    name: str = ""
    source_language: str = ""
    target_language: str = ""
    file_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    translation_stats: ProjectTranslationStats = field(
        default_factory=ProjectTranslationStats
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProjectPreviewPayload":
        """把数据层返回的工程摘要字典规范化为响应载荷对象。"""

        normalized: dict[str, Any]
        if isinstance(data, dict):
            normalized = data
        else:
            normalized = {}

        translation_stats = ProjectTranslationStats.from_dict(
            normalized.get("translation_stats"),
        )

        return cls(
            path=str(normalized.get("path", "")),
            name=str(normalized.get("name", "")),
            source_language=str(normalized.get("source_language", "")),
            target_language=str(normalized.get("target_language", "")),
            file_count=int(normalized.get("file_count", 0) or 0),
            created_at=str(normalized.get("created_at", "")),
            updated_at=str(normalized.get("updated_at", "")),
            translation_stats=translation_stats,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为稳定 JSON 结构，供 HTTP 响应载荷使用。"""

        return {
            "path": self.path,
            "name": self.name,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "file_count": self.file_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "translation_stats": self.translation_stats.to_dict(),
        }
