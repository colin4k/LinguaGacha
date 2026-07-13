from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class ProjectSnapshot:
    """工程快照在客户端内冻结，避免页面再推断加载态默认值。"""

    path: str = ""
    loaded: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProjectSnapshot":
        """把工程响应统一转换为稳定的快照对象。"""

        normalized: dict[str, Any]
        if isinstance(data, dict):
            normalized = data
        else:
            normalized = {}

        return cls(
            path=str(normalized.get("path", "")),
            loaded=bool(normalized.get("loaded", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """把工程快照转换回 JSON 字典，供 HTTP 边界复用。"""

        return {
            "path": self.path,
            "loaded": self.loaded,
        }


@dataclass(frozen=True)
class ProjectTranslationStats:
    """工程翻译统计和工作台保持同一四段口径。"""

    total_items: int = 0
    completed_count: int = 0
    failed_count: int = 0
    pending_count: int = 0
    skipped_count: int = 0
    completion_percent: float = 0.0

    @staticmethod
    def normalize_count(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def normalize_percent(value: Any) -> float:
        try:
            return max(0.0, min(100.0, float(value or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "ProjectTranslationStats":
        """把预览统计规范化为复合进度条稳定结构。"""

        if isinstance(data, dict):
            total_items = cls.normalize_count(data.get("total_items"))
            completed_count = cls.normalize_count(data.get("completed_count"))
            failed_count = cls.normalize_count(data.get("failed_count"))
            skipped_count = cls.normalize_count(data.get("skipped_count"))
            default_pending_count = max(
                0,
                total_items - completed_count - failed_count - skipped_count,
            )
            pending_count = cls.normalize_count(
                data.get("pending_count", default_pending_count)
            )
            completion_percent = cls.normalize_percent(data.get("completion_percent"))
            if completion_percent == 0.0 and total_items > 0:
                completion_percent = (
                    (completed_count + skipped_count) / total_items
                ) * 100
            return cls(
                total_items=total_items,
                completed_count=completed_count,
                failed_count=failed_count,
                pending_count=pending_count,
                skipped_count=skipped_count,
                completion_percent=completion_percent,
            )

        return cls()

    def to_dict(self) -> dict[str, int | float]:
        """转换为稳定 JSON 结构，供前端复合进度条消费。"""

        return {
            "total_items": self.total_items,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "pending_count": self.pending_count,
            "skipped_count": self.skipped_count,
            "completion_percent": self.completion_percent,
        }


@dataclass(frozen=True)
class ProjectPreview:
    """工程预览对象显式建模摘要字段，避免页面退回字典式读取。"""

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
    def from_dict(cls, data: dict[str, Any] | None) -> "ProjectPreview":
        """把工程预览响应转换为冻结对象，统一页面消费入口。"""

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
        """把预览对象转换回显式摘要字典，避免泄漏未建模字段。"""

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
