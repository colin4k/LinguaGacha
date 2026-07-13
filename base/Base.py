from enum import StrEnum
from typing import Callable

from base.EventManager import EventManager


class Base:
    APP_NAME: str = "LinguaGacha"
    APP_VERSION: str = "0.0.0"
    REPO_URL: str = "https://github.com/neavo/LinguaGacha"
    USER_AGENT_NAME: str = "LinguaGacha"

    # 翻译/分析事件速查表：
    # +-------------------------------+-------------------------------+------------------------------------------------+-----------------------------------------------------------+
    # | 事件名                        | sub_event                     | 语义                                           | 常见字段                                                  |
    # +-------------------------------+-------------------------------+------------------------------------------------+-----------------------------------------------------------+
    # | TRANSLATION_TASK              | REQUEST / RUN / DONE / ERROR | 发起或继续翻译任务，并回传任务终态            | mode, final_status(SUCCESS/STOPPED/FAILED), message      |
    # | TRANSLATION_REQUEST_STOP      | REQUEST / RUN                | 请求停止当前正在执行的翻译任务（不单独发 DONE）| 无                                                        |
    # | TRANSLATION_PROGRESS          | （按快照事件处理）           | 上报翻译进度快照                               | line, total_line, processed_line, error_line, total_tokens, time |
    # | ANALYSIS_TASK                 | REQUEST / RUN / DONE / ERROR | 发起或继续术语分析任务，并回传任务终态         | mode, final_status(SUCCESS/STOPPED/FAILED), message      |
    # | ANALYSIS_REQUEST_STOP         | REQUEST / RUN                | 请求停止当前正在执行的分析任务（不单独发 DONE）| 无                                                        |
    # | ANALYSIS_PROGRESS             | （按快照事件处理）           | 上报分析进度快照                               | line, total_line, processed_line, error_line, total_tokens, time |
    # | RETRANSLATE_TASK              | REQUEST / RUN / DONE / ERROR | 发起或完成批量重翻任务                       | item_ids, final_status(SUCCESS/FAILED), message          |
    # | PROJECT_RUNTIME_PATCH         | UPDATE                       | 直接推送 V2 运行态补丁                        | updatedSections, patch, sectionRevisions, projectRevision |
    # +-------------------------------+-------------------------------+------------------------------------------------+-----------------------------------------------------------+

    # 事件
    class Event(StrEnum):
        TRANSLATION_TASK = "TRANSLATION_TASK"  # 翻译 - 任务生命周期事件
        TRANSLATION_REQUEST_STOP = "TRANSLATION_REQUEST_STOP"  # 翻译 - 停止请求
        TRANSLATION_PROGRESS = "TRANSLATION_PROGRESS"  # 翻译 - 进度快照
        TRANSLATION_EXPORT = "TRANSLATION_EXPORT"  # 翻译 - 导出
        TRANSLATION_MANUAL_EXPORT = "TRANSLATION_MANUAL_EXPORT"  # 翻译 - 手动导出
        TRANSLATION_UPDATE = "TRANSLATION_UPDATE"  # 翻译 - 更新（旧名）
        TRANSLATION_START = "TRANSLATION_START"  # 翻译 - 开始（旧名）
        TRANSLATION_STOP = "TRANSLATION_STOP"  # 翻译 - 停止（旧名）
        TRANSLATION_STOP_DONE = "TRANSLATION_STOP_DONE"  # 翻译 - 停止完成（旧名）
        TRANSLATION_RESET_ALL = "TRANSLATION_RESET_ALL"  # 翻译 - 全部重置
        TRANSLATION_RESET_FAILED = "TRANSLATION_RESET_FAILED"  # 翻译 - 重置失败
        ANALYSIS_TASK = "ANALYSIS_TASK"  # 分析 - 任务生命周期事件
        ANALYSIS_REQUEST_STOP = "ANALYSIS_REQUEST_STOP"  # 分析 - 停止请求
        ANALYSIS_PROGRESS = "ANALYSIS_PROGRESS"  # 分析 - 进度快照
        RETRANSLATE_TASK = "RETRANSLATE_TASK"  # 重翻 - 任务生命周期事件
        PROJECT_LOADED = "PROJECT_LOADED"  # 工程 - 已加载
        PROJECT_UNLOADED = "PROJECT_UNLOADED"  # 工程 - 已卸载
        PROJECT_RUNTIME_PATCH = "PROJECT_RUNTIME_PATCH"  # 工程 - 运行态补丁
        PROJECT_CHECK = "PROJECT_CHECK"  # 工程 - 检查
        PROJECT_FILE_UPDATE = "PROJECT_FILE_UPDATE"  # 工程 - 文件更新
        PROJECT_PREFILTER = "PROJECT_PREFILTER"  # 工程 - 预过滤
        CONFIG_UPDATED = "CONFIG_UPDATED"  # 配置 - 已更新
        QUALITY_RULE_UPDATE = "QUALITY_RULE_UPDATE"  # 质量规则 - 更新
        WORKBENCH_REFRESH = "WORKBENCH_REFRESH"  # 工作台 - 刷新
        WORKBENCH_SNAPSHOT = "WORKBENCH_SNAPSHOT"  # 工作台 - 快照
        APITEST = "APITEST"  # API 测试
        APP_UPDATE_APPLY = "APP_UPDATE_APPLY"  # 应用更新 - 应用
        APP_UPDATE_CHECK = "APP_UPDATE_CHECK"  # 应用更新 - 检查
        APP_UPDATE_DOWNLOAD = "APP_UPDATE_DOWNLOAD"  # 应用更新 - 下载
        TOAST = "TOAST"  # 提示框
        PROGRESS_TOAST = "PROGRESS_TOAST"  # 进度提示框

    # 通用生命周期子事件
    # 为什么需要它：多数事件都遵循“请求 -> 运行 -> 更新 -> 完成/失败”的同构流程，
    # 统一枚举能减少 if-else 分叉并保持 payload 结构稳定。
    class SubEvent(StrEnum):
        REQUEST = "REQUEST"  # 请求阶段
        RUN = "RUN"  # 执行阶段
        UPDATE = "UPDATE"  # 中间进度更新阶段
        DONE = "DONE"  # 成功完成阶段
        ERROR = "ERROR"  # 失败终态阶段

    # 接口格式
    class APIFormat(StrEnum):
        OPENAI = "OpenAI"
        GOOGLE = "Google"
        ANTHROPIC = "Anthropic"
        SAKURALLM = "SakuraLLM"

    # 任务类型
    class TaskType(StrEnum):
        NER = "NER"
        TRANSLATION = "TRANSLATION"
        ANALYSIS = "ANALYSIS"

    # 任务状态
    class TaskStatus(StrEnum):
        IDLE = "IDLE"  # 无任务
        ANALYZING = "ANALYZING"  # 分析中
        TRANSLATING = "TRANSLATING"  # 翻译中
        RETRANSLATING = "RETRANSLATING"  # 重翻中
        STOPPING = "STOPPING"  # 停止中

    # 条目状态
    class ItemStatus(StrEnum):
        NONE = "NONE"  # 无
        PROCESSED = "PROCESSED"  # 已处理
        PROCESSING = "PROCESSING"  # 处理中（旧名，向后兼容）
        PROCESSED_IN_PAST = "PROCESSED_IN_PAST"  # 历史已处理（旧名，向后兼容）
        EXCLUDED = "EXCLUDED"  # 已排除
        RULE_SKIPPED = "RULE_SKIPPED"  # 规则跳过
        LANGUAGE_SKIPPED = "LANGUAGE_SKIPPED"  # 非目标原文语言
        DUPLICATED = "DUPLICATED"  # 重复条目
        ERROR = "ERROR"  # 处理出错/重试失败

    # 向后兼容别名：旧代码使用 Base.ProjectStatus，新代码使用 Base.ItemStatus
    ProjectStatus = ItemStatus

    # 提示框类型（GUI 层使用）
    class ToastType(StrEnum):
        INFO = "INFO"
        SUCCESS = "SUCCESS"
        WARNING = "WARNING"
        ERROR = "ERROR"

    # 工程预过滤子事件（GUI 层使用）
    class ProjectPrefilterSubEvent(StrEnum):
        RUN = "RUN"
        UPDATED = "UPDATED"
        DONE = "DONE"
        ERROR = "ERROR"

    # 翻译状态（api_service.py 使用，向后兼容）
    class TranslationStatus(StrEnum):
        UNTRANSLATED = "UNTRANSLATED"
        TRANSLATING = "TRANSLATING"

    # API 服务工作状态（api_service.py 使用，向后兼容；注意 STOPING 拼写保持原样）
    class Status(StrEnum):
        IDLE = "IDLE"
        TRANSLATING = "TRANSLATING"
        STOPING = "STOPING"

    WORK_STATUS: "Base.Status" = None  # type: ignore[assignment]  # initialized at runtime by api_service.py

    # 翻译模式 (用户意图)
    class TranslationMode(StrEnum):
        NEW = "NEW"  # 新任务：从工程数据库加载条目，并初始化全新进度
        CONTINUE = "CONTINUE"  # 继续翻译：从工程数据库加载条目，并恢复既有进度
        RESET = "RESET"  # 重置任务 (强制重解析 Assets)

    # 分析模式 (用户意图)
    class AnalysisMode(StrEnum):
        NEW = "NEW"  # 新任务：清空既有分析进度并重新扫描全部文件
        CONTINUE = "CONTINUE"  # 继续分析：跳过已完成文件，仅继续剩余文件
        RESET = "RESET"  # 重置任务：用于外部显式声明“重新构建分析语料”

    API_STREAM_SOURCE_EVENTS: tuple[Event, ...] = (
        Event.PROJECT_LOADED,
        Event.PROJECT_UNLOADED,
        Event.PROJECT_RUNTIME_PATCH,
        Event.TRANSLATION_TASK,
        Event.TRANSLATION_REQUEST_STOP,
        Event.TRANSLATION_PROGRESS,
        Event.ANALYSIS_TASK,
        Event.ANALYSIS_REQUEST_STOP,
        Event.ANALYSIS_PROGRESS,
        Event.RETRANSLATE_TASK,
        Event.CONFIG_UPDATED,
    )
    ENGINE_BUSY_STATUSES: tuple[TaskStatus, ...] = (
        TaskStatus.TRANSLATING,
        TaskStatus.ANALYZING,
        TaskStatus.RETRANSLATING,
        TaskStatus.STOPPING,
    )

    # 构造函数
    # Base 作为 mixin 使用：统一暴露应用级事件总线入口，
    # 不再依赖任何 Qt 对象或主线程语义。
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)

    # 触发事件
    def emit(self, signal: object, *args: object) -> bool:
        """统一的 emit 入口。

        说明：Base 统一拦截应用事件；若上层类也定义了 emit，则继续委派，
        这样不同宿主对象仍能复用同一份 mixin 入口：

        - 若 signal 是 Base.Event：走应用事件总线
        - 否则：尝试委派给父类 emit
        """

        if isinstance(signal, Base.Event):
            payload = args[0] if args else {}
            payload_dict = payload if isinstance(payload, dict) else {}
            EventManager.get().emit_event(signal, payload_dict)
            return True

        super_emit = getattr(super(), "emit", None)
        if callable(super_emit):
            return bool(super_emit(signal, *args))
        return False

    # 订阅事件
    def subscribe(self, event: Event, hanlder: Callable) -> None:
        EventManager.get().subscribe(event, hanlder)

    # 取消订阅事件
    def unsubscribe(self, event: Event, hanlder: Callable) -> None:
        EventManager.get().unsubscribe(event, hanlder)

    @classmethod
    def is_engine_busy(cls, status: TaskStatus) -> bool:
        """统一定义哪些引擎状态需要锁住会影响任务语义的控件。"""

        return status in cls.ENGINE_BUSY_STATUSES
