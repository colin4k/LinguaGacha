import os
import sys
import time
import base64
import shutil
import threading
import tempfile
from typing import Dict, List, Optional, Any, Union

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Query, Body, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles
import uvicorn

# 导入原有应用的模块
from base.Base import Base
from module.Translator.Translator import Translator
from module.File.FileManager import FileManager
from module.File.FileChecker import FileChecker
from module.Platform.PlatformTester import PlatformTester
from module.Localizer.Localizer import Localizer
from module.LogHelper import LogHelper
from module.Cache.CacheManager import CacheManager
from module.Cache.CacheItem import CacheItem

# 创建FastAPI应用
app = FastAPI(
    title="LinguaGacha API",
    description="使用 AI 能力一键翻译多种格式文本内容的次世代文本翻译器 API，支持字幕(.srt .ass)、电子书(.txt .epub)、Markdown(.md)、RenPy(.rpy)、MTool(.json)、SExtractor(.txt .json .xlsx)、VNTextPatch(.json)、Translator++(.trans .xlsx)、WOLF官方翻译工具(.xlsx)等格式",
    version="1.0.0",
    docs_url=None
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
translator = Translator()
platform_tester = PlatformTester()
cache_manager = CacheManager()
translation_tasks = {}

# 挂载静态文件路径
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Custom Swagger UI",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css"
    )

# 载入配置文件
def load_config() -> dict:
    config = {}
    if os.path.exists("resource/config.json"):
        with open("resource/config.json", "r", encoding="utf-8-sig") as reader:
            import rapidjson as json
            config = json.load(reader)
    return config

# 初始化目录
os.makedirs("./input", exist_ok=True)
os.makedirs("./output", exist_ok=True)

# 模型定义
class ApiConfig(BaseModel):
    # 应用基本配置
    app_language: Optional[str] = Field(default="ZH", description="应用语言(ZH/EN/JA)")
    theme: Optional[str] = Field(default="light", description="应用主题(light/dark)")

    # 翻译配置
    source_language: Optional[str] = Field(default="EN", description="源语言")
    target_language: Optional[str] = Field(default="ZH", description="目标语言")
    input_folder: Optional[str] = Field(default="./input", description="输入文件夹路径")
    output_folder: Optional[str] = Field(default="./output", description="输出文件夹路径")
    traditional_chinese_enable: Optional[bool] = Field(default=False, description="是否启用繁体中文")
    task_token_limit: Optional[int] = Field(default=384, description="翻译任务长度阈值")
    batch_size: Optional[int] = Field(default=4, description="并发任务数")
    request_timeout: Optional[int] = Field(default=60, description="请求超时时间(秒)")
    max_round: Optional[int] = Field(default=16, description="最大轮数")

    # 词汇表和替换配置
    auto_glossary_enable: Optional[bool] = Field(default=False, description="是否启用自动词汇表")
    mtool_optimizer_enable: Optional[bool] = Field(default=False, description="是否启用MTool优化器")
    glossary_enable: Optional[bool] = Field(default=False, description="是否启用词汇表")
    glossary_data: Optional[List[Dict[str, str]]] = Field(default=[], description="词汇表数据")
    pre_translation_replacement_enable: Optional[bool] = Field(default=False, description="是否启用翻译前替换")
    pre_translation_replacement_data: Optional[List[Dict[str, str]]] = Field(default=[], description="翻译前替换数据")
    post_translation_replacement_enable: Optional[bool] = Field(default=False, description="是否启用翻译后替换")
    post_translation_replacement_data: Optional[List[Dict[str, str]]] = Field(default=[], description="翻译后替换数据")

    # 翻译结果检查配置
    kana_check_enable: Optional[bool] = Field(default=True, description="是否启用假名残留检查")
    hangeul_check_enable: Optional[bool] = Field(default=True, description="是否启用韩文残留检查")
    code_check_enable: Optional[bool] = Field(default=True, description="是否启用代码残留检查")
    similarity_check_enable: Optional[bool] = Field(default=True, description="是否启用相似度检查")
    glossary_check_enable: Optional[bool] = Field(default=True, description="是否启用词汇表检查")
    untranslated_check_enable: Optional[bool] = Field(default=True, description="是否启用未翻译检查")
    retry_count_threshold_check_enable: Optional[bool] = Field(default=True, description="是否启用重试次数阈值检查")

    # 代理配置
    proxy_enable: Optional[bool] = Field(default=False, description="是否启用代理")
    proxy_url: Optional[str] = Field(default="", description="代理URL")

    # 平台配置 (单独的API用于平台配置管理)
    activate_platform: Optional[int] = Field(default=0, description="当前激活的平台ID")

class TranslationResponse(BaseModel):
    task_id: str
    status: str
    message: str

class PlatformConfig(BaseModel):
    id: int = Field(default=0, description="平台ID")
    name: str = Field(default="OpenAI", description="平台名称")
    api_url: str = Field(default="https://api.openai.com/v1", description="API URL")
    api_key: List[str] = Field(default=[], description="API密钥列表")
    api_format: str = Field(default="OpenAI", description="API格式，支持OpenAI/Google/Anthropic/SakuraLLM")
    model: str = Field(default="gpt-4o-mini", description="模型名称")
    thinking: Optional[bool] = Field(default=False, description="是否启用思考模式")
    think: Optional[bool] = Field(default=False, description="是否启用思考模式(兼容旧版配置)")
    top_p: float = Field(default=0.95, description="top_p参数")
    temperature: float = Field(default=0.75, description="温度参数")
    presence_penalty: float = Field(default=0.0, description="存在惩罚参数")
    frequency_penalty: float = Field(default=0.0, description="频率惩罚参数")

class PlatformResponse(BaseModel):
    status: bool
    message: str
    platform_info: Dict = {}

class TranslationStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    total_items: int
    translated_items: int
    message: str

class FilesListResponse(BaseModel):
    input_files: List[str]
    output_files: List[str]

class PlatformsListResponse(BaseModel):
    platforms: List[PlatformConfig]
    activate_platform: int

# 初始化路由
@app.get("/", tags=["状态"])
def get_root():
    return {"message": "LinguaGacha API 服务正在运行", "version": "1.0.0"}

@app.post("/api/config", tags=["配置"])
def update_config(config: ApiConfig):
    """
    更新翻译配置
    """
    try:
        # 获取当前配置
        current_config = load_config()

        # 更新配置
        for key, value in config.dict(exclude_unset=True).items():
            current_config[key] = value

        # 保存配置
        if not os.path.exists("resource"):
            os.makedirs("resource", exist_ok=True)

        with open("resource/config.json", "w", encoding="utf-8-sig") as writer:
            import rapidjson as json
            json.dump(current_config, writer, indent=4)

        return {"status": "success", "message": "配置已更新"}
    except Exception as e:
        LogHelper.error(f"更新配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")

@app.get("/api/config", tags=["配置"])
def get_config():
    """
    获取当前翻译配置
    """
    try:
        return load_config()
    except Exception as e:
        LogHelper.error(f"获取配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")

@app.post("/api/platform/test", tags=["平台"])
def test_platform(platform_config: PlatformConfig):
    """
    测试API平台连接
    """
    try:
        # 转换为平台测试器需要的格式
        test_config = {
            "api_url": platform_config.api_url,
            "api_key": platform_config.api_key,
            "api_format": platform_config.api_format,
            "model": platform_config.model,
            "thinking": platform_config.thinking or platform_config.think,
            "temperature": platform_config.temperature,
            "top_p": platform_config.top_p,
            "presence_penalty": platform_config.presence_penalty,
            "frequency_penalty": platform_config.frequency_penalty
        }

        result = platform_tester.test(test_config)
        return PlatformResponse(
            status=result.get("status", False),
            message=result.get("message", ""),
            platform_info=result.get("platform", {})
        )
    except Exception as e:
        LogHelper.error(f"测试平台失败: {str(e)}")
        return PlatformResponse(
            status=False,
            message=f"测试平台失败: {str(e)}",
            platform_info={}
        )

@app.post("/api/files/upload", tags=["文件"])
async def upload_file(file: UploadFile = File(...), folder: str = Form("input")):
    """
    上传文件到指定文件夹
    """
    try:
        # 验证目标文件夹
        valid_folders = ["input"]
        if folder not in valid_folders:
            raise HTTPException(status_code=400, detail=f"无效的目标文件夹，有效值为: {', '.join(valid_folders)}")

        # 保存文件
        file_path = os.path.join(folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"status": "success", "message": f"文件已上传到 {file_path}"}
    except HTTPException as e:
        raise e
    except Exception as e:
        LogHelper.error(f"上传文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传文件失败: {str(e)}")

@app.get("/api/files/list", tags=["文件"])
def list_files():
    """
    列出输入和输出文件夹中的文件
    """
    try:
        input_files = []
        output_files = []

        # 列出输入文件夹中的文件
        if os.path.exists("input"):
            for root, _, files in os.walk("input"):
                for file in files:
                    input_files.append(os.path.join(root, file).replace("\\", "/"))

        # 列出输出文件夹中的文件
        if os.path.exists("output"):
            for root, _, files in os.walk("output"):
                for file in files:
                    output_files.append(os.path.join(root, file).replace("\\", "/"))

        return FilesListResponse(input_files=input_files, output_files=output_files)
    except Exception as e:
        LogHelper.error(f"列出文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"列出文件失败: {str(e)}")

@app.get("/api/files/download/{file_path:path}", tags=["文件"])
def download_file(file_path: str):
    """
    下载指定路径的文件
    """
    try:
        # 安全检查，防止路径遍历攻击
        norm_path = os.path.normpath(file_path)
        if norm_path.startswith("..") or norm_path.startswith("/"):
            raise HTTPException(status_code=403, detail="非法的文件路径")

        # 检查文件是否存在
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail=f"文件 {file_path} 不存在")

        return FileResponse(path=file_path, filename=os.path.basename(file_path))
    except HTTPException as e:
        raise e
    except Exception as e:
        LogHelper.error(f"下载文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"下载文件失败: {str(e)}")

@app.delete("/api/files/delete/{file_path:path}", tags=["文件"])
def delete_file(file_path: str):
    """
    删除指定路径的文件
    """
    try:
        # 安全检查，防止路径遍历攻击
        norm_path = os.path.normpath(file_path)
        if norm_path.startswith("..") or norm_path.startswith("/"):
            raise HTTPException(status_code=403, detail="非法的文件路径")

        # 检查文件是否存在
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail=f"文件 {file_path} 不存在")

        # 删除文件
        os.remove(file_path)
        return {"status": "success", "message": f"文件 {file_path} 已删除"}
    except HTTPException as e:
        raise e
    except Exception as e:
        LogHelper.error(f"删除文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")

@app.post("/api/translation/start", tags=["翻译"])
def start_translation(background_tasks: BackgroundTasks):
    """
    开始翻译任务
    """
    try:
        # 生成任务ID
        import uuid
        task_id = str(uuid.uuid4())

        # 设置初始任务状态
        translation_tasks[task_id] = {
            "status": "initializing",
            "progress": 0.0,
            "total_items": 0,
            "translated_items": 0,
            "message": "正在初始化翻译任务..."
        }

        # 在后台运行翻译任务
        background_tasks.add_task(translation_task, task_id)

        return TranslationResponse(
            task_id=task_id,
            status="initializing",
            message="翻译任务已开始初始化"
        )
    except Exception as e:
        LogHelper.error(f"启动翻译任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动翻译任务失败: {str(e)}")

@app.get("/api/translation/status/{task_id}", tags=["翻译"])
def get_translation_status(task_id: str):
    """
    获取翻译任务状态
    """
    try:
        if task_id not in translation_tasks:
            raise HTTPException(status_code=404, detail=f"未找到ID为 {task_id} 的翻译任务")

        task_info = translation_tasks[task_id]

        return TranslationStatusResponse(
            task_id=task_id,
            status=task_info.get("status", "unknown"),
            progress=task_info.get("progress", 0.0),
            total_items=task_info.get("total_items", 0),
            translated_items=task_info.get("translated_items", 0),
            message=task_info.get("message", "")
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        LogHelper.error(f"获取翻译状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取翻译状态失败: {str(e)}")

@app.post("/api/translation/stop/{task_id}", tags=["翻译"])
def stop_translation(task_id: str):
    """
    停止翻译任务
    """
    try:
        if task_id not in translation_tasks:
            raise HTTPException(status_code=404, detail=f"未找到ID为 {task_id} 的翻译任务")

        # 设置停止标志
        translation_tasks[task_id]["status"] = "stopping"
        translation_tasks[task_id]["message"] = "正在停止翻译任务..."

        # 通知应用停止翻译
        translator.emit(Base.Event.TRANSLATION_STOP, {})

        return {"status": "success", "message": "已发送停止翻译请求"}
    except HTTPException as e:
        raise e
    except Exception as e:
        LogHelper.error(f"停止翻译任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"停止翻译任务失败: {str(e)}")

@app.post("/api/translation/export/{task_id}", tags=["翻译"])
def export_translation(task_id: str):
    """
    手动导出翻译结果
    """
    try:
        if task_id not in translation_tasks:
            raise HTTPException(status_code=404, detail=f"未找到ID为 {task_id} 的翻译任务")

        # 检查任务状态
        task_info = translation_tasks[task_id]
        if task_info.get("status") != "translating":
            raise HTTPException(status_code=400, detail="只有在翻译中的任务才能手动导出")

        # 通知应用导出翻译结果
        translator.emit(Base.Event.TRANSLATION_MANUAL_EXPORT, {})

        return {"status": "success", "message": "已发送导出翻译请求"}
    except HTTPException as e:
        raise e
    except Exception as e:
        LogHelper.error(f"导出翻译结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"导出翻译结果失败: {str(e)}")

@app.post("/api/translation/continue", tags=["翻译"])
def continue_translation(background_tasks: BackgroundTasks):
    """
    继续上一次的翻译任务
    """
    try:
        # 检查是否有保存的翻译项目
        config = load_config()
        output_folder = config.get("output_folder", "./output")
        cache_manager_temp = CacheManager()

        if not cache_manager_temp.load_project_from_file(output_folder):
            raise HTTPException(status_code=404, detail="没有找到可继续的翻译项目")

        if cache_manager_temp.get_project().get_status() != Base.TranslationStatus.TRANSLATING:
            raise HTTPException(status_code=400, detail="没有处于翻译中状态的项目可继续")

        # 生成任务ID
        import uuid
        task_id = str(uuid.uuid4())

        # 设置初始任务状态
        translation_tasks[task_id] = {
            "status": "initializing",
            "progress": 0.0,
            "total_items": 0,
            "translated_items": 0,
            "message": "正在初始化继续翻译任务..."
        }

        # 在后台运行翻译任务（继续模式）
        background_tasks.add_task(continue_translation_task, task_id)

        return TranslationResponse(
            task_id=task_id,
            status="initializing",
            message="继续翻译任务已开始初始化"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        LogHelper.error(f"启动继续翻译任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动继续翻译任务失败: {str(e)}")

# 翻译任务后台处理函数
def translation_task(task_id: str):
    try:
        global cache_manager

        LogHelper.info(f"========== 开始翻译任务 {task_id} ==========")
        LogHelper.info(f"翻译任务开始前的全局状态: WORK_STATUS={Base.WORK_STATUS}")
        LogHelper.info(f"全局translator对象: id={id(translator)}, 类型={type(translator)}")
        LogHelper.info(f"全局cache_manager对象: id={id(cache_manager)}, 类型={type(cache_manager)}")

        # 检查translator内部状态
        if hasattr(translator, 'translating'):
            LogHelper.info(f"翻译器内部状态: translating={translator.translating}")
        else:
            LogHelper.info("翻译器没有translating属性")

        # 检查事件处理器注册情况
        if hasattr(translator, '_handlers'):
            for event, handlers in translator._handlers.items():
                if event == Base.Event.TRANSLATION_START:
                    LogHelper.info(f"TRANSLATION_START事件处理器: {len(handlers)}个")
                    for i, handler in enumerate(handlers):
                        LogHelper.info(f"  处理器{i}: {handler}")

        # 设置全局状态为翻译中
        Base.WORK_STATUS = Base.Status.TRANSLATING
        LogHelper.info(f"设置全局状态: WORK_STATUS={Base.WORK_STATUS}")

        # 更新任务状态
        translation_tasks[task_id].update({
            "status": "translating",
            "message": "翻译任务进行中..."
        })

        # 创建自定义的进度回调
        def progress_callback(event: int, data: dict):
            LogHelper.info(f"收到翻译进度更新: event={event}, 任务ID={task_id}")
            if task_id in translation_tasks and translation_tasks[task_id]["status"] == "translating":
                # 从事件数据中提取进度信息
                if "total_line" in data and "line" in data:
                    total = data.get("total_line", 0)
                    done = data.get("line", 0)

                    if total > 0:
                        progress = min(1.0, done / total)

                        translation_tasks[task_id].update({
                            "progress": progress,
                            "total_items": total,
                            "translated_items": done,
                            "message": f"已完成 {done}/{total} 个项目"
                        })
                        LogHelper.info(f"更新翻译进度: {done}/{total}, 进度={progress:.2f}")

        # 创建翻译完成回调
        def complete_callback(event: int, data: dict):
            LogHelper.info(f"收到翻译完成事件: TRANSLATION_STOP_DONE, 任务ID={task_id}, 全局状态={Base.WORK_STATUS}")
            # 任务已经被标记为停止时，不再更新状态
            if task_id in translation_tasks and translation_tasks[task_id]["status"] not in ["completed", "failed"]:
                # 检查全局状态来区分自然完成和手动停止
                if Base.WORK_STATUS == Base.Status.STOPING:
                    # 如果全局状态是停止中，说明是用户手动停止
                    translation_tasks[task_id].update({
                        "status": "stopped",
                        "message": "翻译任务已被用户停止"
                    })
                    LogHelper.info(f"任务状态更新为stopped: {task_id}")
                else:
                    # 否则是自然完成
                    translation_tasks[task_id].update({
                        "status": "completed",
                        "progress": 1.0,
                        "message": "翻译任务已成功完成"
                    })
                    LogHelper.info(f"任务状态更新为completed: {task_id}")

            # 取消订阅，避免内存泄漏
            translator.unsubscribe(Base.Event.TRANSLATION_STOP_DONE, complete_callback)
            translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
            LogHelper.info("已取消事件订阅")

        # 注册事件回调
        LogHelper.info(f"注册翻译事件处理器，任务ID: {task_id}")
        translator.subscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
        translator.subscribe(Base.Event.TRANSLATION_STOP_DONE, complete_callback)

        # 加载配置
        config = load_config()

        # 确保输入输出文件夹存在
        input_folder = config.get("input_folder", "./input")
        output_folder = config.get("output_folder", "./output")
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

        # 检查输入文件夹中是否有文件
        files = []
        for root, _, filenames in os.walk(input_folder):
            for filename in filenames:
                files.append(os.path.join(root, filename))

        if not files:
            LogHelper.error("输入文件夹中没有找到文件，无法启动翻译")
            translation_tasks[task_id].update({
                "status": "failed",
                "message": "输入文件夹中没有找到文件，无法启动翻译"
            })
            # 取消订阅
            translator.unsubscribe(Base.Event.TRANSLATION_STOP_DONE, complete_callback)
            translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
            return

        LogHelper.info(f"找到输入文件 {len(files)} 个: {files}")

        # 准备翻译任务参数，确保与原始应用一致
        translation_params = {
            "status": Base.TranslationStatus.UNTRANSLATED,
            "source_language": config.get("source_language", "EN"),
            "target_language": config.get("target_language", "ZH"),
            "input_folder": input_folder,
            "output_folder": output_folder,
            "batch_size": config.get("batch_size", 4),
            "request_timeout": config.get("request_timeout", 60),
            "task_token_limit": config.get("task_token_limit", 384),
            "max_round": config.get("max_round", 16),
            "traditional_chinese_enable": config.get("traditional_chinese_enable", False),
            "auto_glossary_enable": config.get("auto_glossary_enable", False),
            "mtool_optimizer_enable": config.get("mtool_optimizer_enable", False),
            "glossary_enable": config.get("glossary_enable", False),
            "glossary_data": config.get("glossary_data", []),
            "pre_translation_replacement_enable": config.get("pre_translation_replacement_enable", False),
            "pre_translation_replacement_data": config.get("pre_translation_replacement_data", []),
            "post_translation_replacement_enable": config.get("post_translation_replacement_enable", False),
            "post_translation_replacement_data": config.get("post_translation_replacement_data", []),
            "kana_check_enable": config.get("kana_check_enable", True),
            "hangeul_check_enable": config.get("hangeul_check_enable", True),
            "code_check_enable": config.get("code_check_enable", True),
            "similarity_check_enable": config.get("similarity_check_enable", True),
            "glossary_check_enable": config.get("glossary_check_enable", True),
            "untranslated_check_enable": config.get("untranslated_check_enable", True),
            "retry_count_threshold_check_enable": config.get("retry_count_threshold_check_enable", True)
        }

        # 将全局缓存管理器重置并设置为与API一致的状态
        # 这确保translator内部使用的cache_manager与我们的预期一致
        cache_manager = CacheManager()
        LogHelper.info(f"重置全局cache_manager: id={id(cache_manager)}")

        # 记录当前translator的事件处理器状态
        LogHelper.info(f"触发翻译前translator事件处理器状态:")
        if hasattr(translator, '_handlers'):
            for event, handlers in list(translator._handlers.items()):
                LogHelper.info(f"  事件 {event}: {len(handlers)} 个处理器")

        # 触发翻译开始事件，传递完整配置
        LogHelper.info(f"触发翻译开始事件: TRANSLATION_START，任务ID: {task_id}")
        translator.emit(Base.Event.TRANSLATION_START, translation_params)

        # 添加状态变化监控
        LogHelper.info(f"事件触发后立即检查: WORK_STATUS={Base.WORK_STATUS}")

        # 除了发送事件，直接调用translation_start方法
        if hasattr(translator, 'translation_start'):
            LogHelper.info(f"直接调用translator.translation_start()方法启动翻译")
            try:
                # 传递正确的参数：事件ID和数据
                event_id = Base.Event.TRANSLATION_START  # 使用TRANSLATION_START事件ID
                translator.translation_start(event_id, translation_params)
                LogHelper.info("translation_start方法调用成功")
            except Exception as e:
                LogHelper.error(f"调用translation_start方法失败: {str(e)}")
                import traceback
                LogHelper.error(f"异常堆栈: {traceback.format_exc()}")

                # 尝试查找translation_start方法的参数信息
                import inspect
                if hasattr(inspect, 'signature') and callable(translator.translation_start):
                    try:
                        sig = inspect.signature(translator.translation_start)
                        LogHelper.info(f"translation_start方法签名: {sig}")
                    except Exception as ex:
                        LogHelper.error(f"无法获取方法签名: {str(ex)}")

        # 额外诊断 - 检查translator实例的属性和方法
        LogHelper.info("进行额外的translator诊断...")
        translator_methods = [method for method in dir(translator) if not method.startswith('_')]
        LogHelper.info(f"Translator可用方法: {translator_methods}")

        # 检查是否有start或run方法
        if hasattr(translator, 'start'):
            LogHelper.info("找到translator.start方法，尝试调用")
            try:
                # 添加标志以防止调用时可能导致的问题
                translator.start()
                LogHelper.info("translator.start()调用成功")
            except Exception as e:
                LogHelper.error(f"调用translator.start()失败: {str(e)}")

        # 检查translator内部状态
        for attr in ['translating', 'status', 'current_task', 'is_running']:
            if hasattr(translator, attr):
                try:
                    val = getattr(translator, attr)
                    LogHelper.info(f"translator.{attr} = {val}")
                except Exception as e:
                    LogHelper.error(f"获取translator.{attr}失败: {str(e)}")

        # 检查是否有通过其他方式启动翻译的方法
        for method_name in ['translate', 'run', 'process', 'translate_file']:
            if hasattr(translator, method_name):
                LogHelper.info(f"找到可能的启动方法: translator.{method_name}")

        # 延迟检查以验证事件处理
        def delayed_status_check():
            for i in range(1, 6):  # 检查5次，每次间隔1秒
                time.sleep(1)
                work_status = Base.WORK_STATUS
                # 检查translator内部状态
                translator_status = "未知"
                if hasattr(translator, 'translating'):
                    translator_status = f"translating={translator.translating}"

                LogHelper.info(f"事件触发后{i}秒: WORK_STATUS={work_status}, translator状态={translator_status}")

                # 第5秒时做额外检查
                if i == 5:
                    LogHelper.info("5秒检查点:")
                    if task_id in translation_tasks:
                        LogHelper.info(f"任务状态: {translation_tasks[task_id]}")
                    if hasattr(translator, '_handlers'):
                        for event, handlers in list(translator._handlers.items()):
                            LogHelper.info(f"  事件 {event}: {len(handlers)} 个处理器")

        # 启动延迟检查线程
        threading.Thread(target=delayed_status_check, daemon=True).start()

        LogHelper.info("翻译任务初始化完成，等待事件系统处理")

    except Exception as e:
        LogHelper.error(f"翻译任务启动失败: {str(e)}")
        import traceback
        LogHelper.error(f"异常堆栈: {traceback.format_exc()}")
        if task_id in translation_tasks:
            translation_tasks[task_id].update({
                "status": "failed",
                "message": f"翻译任务启动失败: {str(e)}"
            })

        # 确保取消订阅
        try:
            translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
            translator.unsubscribe(Base.Event.TRANSLATION_STOP_DONE, complete_callback)
        except Exception as ex:
            LogHelper.error(f"取消订阅时出错: {str(ex)}")

# 继续翻译任务后台处理函数
def continue_translation_task(task_id: str):
    try:
        global cache_manager

        LogHelper.info(f"========== 开始继续翻译任务 {task_id} ==========")
        LogHelper.info(f"继续翻译任务开始前的全局状态: WORK_STATUS={Base.WORK_STATUS}")
        LogHelper.info(f"全局translator对象: id={id(translator)}, 类型={type(translator)}")
        LogHelper.info(f"全局cache_manager对象: id={id(cache_manager)}, 类型={type(cache_manager)}")

        # 检查translator内部状态
        if hasattr(translator, 'translating'):
            LogHelper.info(f"翻译器内部状态: translating={translator.translating}")
        else:
            LogHelper.info("翻译器没有translating属性")

        # 检查事件处理器注册情况
        if hasattr(translator, '_handlers'):
            for event, handlers in translator._handlers.items():
                if event == Base.Event.TRANSLATION_START:
                    LogHelper.info(f"TRANSLATION_START事件处理器: {len(handlers)}个")
                    for i, handler in enumerate(handlers):
                        LogHelper.info(f"  处理器{i}: {handler}")

        # 设置全局状态为翻译中
        Base.WORK_STATUS = Base.Status.TRANSLATING
        LogHelper.info(f"设置全局状态: WORK_STATUS={Base.WORK_STATUS}")

        # 更新任务状态
        translation_tasks[task_id].update({
            "status": "translating",
            "message": "继续翻译任务进行中..."
        })

        # 创建自定义的进度回调
        def progress_callback(event: int, data: dict):
            LogHelper.info(f"收到翻译进度更新: event={event}, 任务ID={task_id}")
            if task_id in translation_tasks and translation_tasks[task_id]["status"] == "translating":
                # 从事件数据中提取进度信息
                if "total_line" in data and "line" in data:
                    total = data.get("total_line", 0)
                    done = data.get("line", 0)

                    if total > 0:
                        progress = min(1.0, done / total)

                        translation_tasks[task_id].update({
                            "progress": progress,
                            "total_items": total,
                            "translated_items": done,
                            "message": f"已完成 {done}/{total} 个项目"
                        })
                        LogHelper.info(f"更新翻译进度: {done}/{total}, 进度={progress:.2f}")

        # 创建翻译完成回调
        def complete_callback(event: int, data: dict):
            LogHelper.info(f"收到继续翻译完成事件: TRANSLATION_STOP_DONE, 任务ID={task_id}, 全局状态={Base.WORK_STATUS}")
            # 任务已经被标记为停止时，不再更新状态
            if task_id in translation_tasks and translation_tasks[task_id]["status"] not in ["completed", "failed"]:
                # 检查全局状态来区分自然完成和手动停止
                if Base.WORK_STATUS == Base.Status.STOPING:
                    # 如果全局状态是停止中，说明是用户手动停止
                    translation_tasks[task_id].update({
                        "status": "stopped",
                        "message": "继续翻译任务已被用户停止"
                    })
                    LogHelper.info(f"任务状态更新为stopped: {task_id}")
                else:
                    # 否则是自然完成
                    translation_tasks[task_id].update({
                        "status": "completed",
                        "progress": 1.0,
                        "message": "继续翻译任务已成功完成"
                    })
                    LogHelper.info(f"任务状态更新为completed: {task_id}")

            # 取消订阅，避免内存泄漏
            translator.unsubscribe(Base.Event.TRANSLATION_STOP_DONE, complete_callback)
            translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
            LogHelper.info("已取消事件订阅")

        # 注册事件回调
        LogHelper.info(f"注册继续翻译事件处理器，任务ID: {task_id}")
        translator.subscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
        translator.subscribe(Base.Event.TRANSLATION_STOP_DONE, complete_callback)

        # 检查是否有保存的翻译项目
        config = load_config()
        output_folder = config.get("output_folder", "./output")

        # 使用全局缓存管理器，确保与translator使用相同实例
        cache_manager = CacheManager()
        LogHelper.info(f"重置全局cache_manager: id={id(cache_manager)}")

        # 加载项目
        LogHelper.info(f"从 {output_folder} 加载项目")
        if not cache_manager.load_project_from_file(output_folder):
            LogHelper.error("没有找到可继续的翻译项目")
            translation_tasks[task_id].update({
                "status": "failed",
                "message": "没有找到可继续的翻译项目"
            })
            # 取消订阅
            translator.unsubscribe(Base.Event.TRANSLATION_STOP_DONE, complete_callback)
            translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
            return

        project_status = cache_manager.get_project().get_status()
        LogHelper.info(f"项目状态: {project_status}")

        if project_status != Base.TranslationStatus.TRANSLATING:
            LogHelper.error("没有处于翻译中状态的项目可继续")
            translation_tasks[task_id].update({
                "status": "failed",
                "message": "没有处于翻译中状态的项目可继续"
            })
            # 取消订阅
            translator.unsubscribe(Base.Event.TRANSLATION_STOP_DONE, complete_callback)
            translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
            return

        # 确保输入输出文件夹存在
        input_folder = config.get("input_folder", "./input")
        output_folder = config.get("output_folder", "./output")
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

        # 准备翻译任务参数
        translation_params = {
            "status": Base.TranslationStatus.TRANSLATING,  # 这里使用TRANSLATING表示继续翻译
            "source_language": config.get("source_language", "EN"),
            "target_language": config.get("target_language", "ZH"),
            "input_folder": input_folder,
            "output_folder": output_folder,
            "batch_size": config.get("batch_size", 4),
            "request_timeout": config.get("request_timeout", 60),
            "task_token_limit": config.get("task_token_limit", 384),
            "max_round": config.get("max_round", 16),
            "traditional_chinese_enable": config.get("traditional_chinese_enable", False),
            "auto_glossary_enable": config.get("auto_glossary_enable", False),
            "mtool_optimizer_enable": config.get("mtool_optimizer_enable", False),
            "glossary_enable": config.get("glossary_enable", False),
            "glossary_data": config.get("glossary_data", []),
            "pre_translation_replacement_enable": config.get("pre_translation_replacement_enable", False),
            "pre_translation_replacement_data": config.get("pre_translation_replacement_data", []),
            "post_translation_replacement_enable": config.get("post_translation_replacement_enable", False),
            "post_translation_replacement_data": config.get("post_translation_replacement_data", []),
            "kana_check_enable": config.get("kana_check_enable", True),
            "hangeul_check_enable": config.get("hangeul_check_enable", True),
            "code_check_enable": config.get("code_check_enable", True),
            "similarity_check_enable": config.get("similarity_check_enable", True),
            "glossary_check_enable": config.get("glossary_check_enable", True),
            "untranslated_check_enable": config.get("untranslated_check_enable", True),
            "retry_count_threshold_check_enable": config.get("retry_count_threshold_check_enable", True)
        }

        # 记录当前translator的事件处理器状态
        LogHelper.info(f"触发继续翻译前translator事件处理器状态:")
        if hasattr(translator, '_handlers'):
            for event, handlers in list(translator._handlers.items()):
                LogHelper.info(f"  事件 {event}: {len(handlers)} 个处理器")

        # 触发翻译开始事件，传递完整配置，状态设置为TRANSLATING表示继续翻译
        LogHelper.info(f"触发继续翻译开始事件: TRANSLATION_START，任务ID: {task_id}")
        translator.emit(Base.Event.TRANSLATION_START, translation_params)

        # 添加状态变化监控
        LogHelper.info(f"事件触发后立即检查: WORK_STATUS={Base.WORK_STATUS}")

        # 除了发送事件，直接调用translation_start方法继续翻译
        if hasattr(translator, 'translation_start'):
            LogHelper.info(f"直接调用translator.translation_start()方法继续翻译")
            try:
                # 传递正确的参数：事件ID和数据
                event_id = Base.Event.TRANSLATION_START  # 使用TRANSLATION_START事件ID
                # 由于是继续翻译，保持参数中的TRANSLATING状态
                translator.translation_start(event_id, translation_params)
                LogHelper.info("继续翻译的translation_start方法调用成功")
            except Exception as e:
                LogHelper.error(f"调用继续翻译的translation_start方法失败: {str(e)}")
                import traceback
                LogHelper.error(f"异常堆栈: {traceback.format_exc()}")

                # 尝试查找translation_start方法的参数信息
                import inspect
                if hasattr(inspect, 'signature') and callable(translator.translation_start):
                    try:
                        sig = inspect.signature(translator.translation_start)
                        LogHelper.info(f"translation_start方法签名: {sig}")
                    except Exception as ex:
                        LogHelper.error(f"无法获取方法签名: {str(ex)}")

        # 额外诊断 - 检查translator实例的属性和方法
        LogHelper.info("进行额外的translator诊断...")
        translator_methods = [method for method in dir(translator) if not method.startswith('_')]
        LogHelper.info(f"Translator可用方法: {translator_methods}")

        # 检查是否有start或run方法
        if hasattr(translator, 'start'):
            LogHelper.info("找到translator.start方法，尝试调用")
            try:
                # 尝试调用start方法
                translator.start()
                LogHelper.info("translator.start()调用成功")
            except Exception as e:
                LogHelper.error(f"调用translator.start()失败: {str(e)}")

        # 检查continue_translation、resume或restart方法
        for method_name in ['continue_translation', 'resume', 'restart']:
            if hasattr(translator, method_name):
                LogHelper.info(f"找到可能的继续翻译方法: translator.{method_name}")
                try:
                    method = getattr(translator, method_name)
                    LogHelper.info(f"尝试调用 translator.{method_name}()...")
                    method()
                    LogHelper.info(f"translator.{method_name}()调用成功")
                except Exception as e:
                    LogHelper.error(f"调用translator.{method_name}()失败: {str(e)}")

        # 检查translator内部状态
        for attr in ['translating', 'status', 'current_task', 'is_running']:
            if hasattr(translator, attr):
                try:
                    val = getattr(translator, attr)
                    LogHelper.info(f"translator.{attr} = {val}")
                except Exception as e:
                    LogHelper.error(f"获取translator.{attr}失败: {str(e)}")

        # 延迟检查以验证事件处理
        def delayed_status_check():
            for i in range(1, 6):  # 检查5次，每次间隔1秒
                time.sleep(1)
                work_status = Base.WORK_STATUS
                # 检查translator内部状态
                translator_status = "未知"
                if hasattr(translator, 'translating'):
                    translator_status = f"translating={translator.translating}"

                LogHelper.info(f"事件触发后{i}秒: WORK_STATUS={work_status}, translator状态={translator_status}")

                # 第5秒时做额外检查
                if i == 5:
                    LogHelper.info("5秒检查点:")
                    if task_id in translation_tasks:
                        LogHelper.info(f"任务状态: {translation_tasks[task_id]}")
                    if hasattr(translator, '_handlers'):
                        for event, handlers in list(translator._handlers.items()):
                            LogHelper.info(f"  事件 {event}: {len(handlers)} 个处理器")

        # 启动延迟检查线程
        threading.Thread(target=delayed_status_check, daemon=True).start()

        LogHelper.info("继续翻译任务初始化完成，等待事件系统处理")

    except Exception as e:
        LogHelper.error(f"继续翻译任务启动失败: {str(e)}")
        import traceback
        LogHelper.error(f"异常堆栈: {traceback.format_exc()}")
        if task_id in translation_tasks:
            translation_tasks[task_id].update({
                "status": "failed",
                "message": f"继续翻译任务启动失败: {str(e)}"
            })

        # 确保取消订阅
        try:
            translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
            translator.unsubscribe(Base.Event.TRANSLATION_STOP_DONE, complete_callback)
        except Exception as ex:
            LogHelper.error(f"取消订阅时出错: {str(ex)}")

# 翻译完成的回调函数（API全局级别）
def translation_complete_callback(task_id: str, success: bool, message: str):
    """
    当翻译任务完成时被调用的回调函数（全局事件使用）
    """
    if task_id in translation_tasks:
        translation_tasks[task_id].update({
            "status": "completed" if success else "failed",
            "progress": 1.0 if success else translation_tasks[task_id].get("progress", 0.0),
            "message": message
        })

# 连接原有应用的事件（API全局监听）
def connect_events():
    LogHelper.info("=== 注册全局事件处理器 ===")

    # 注册全局事件处理器 - 翻译停止完成
    def on_translation_stop_done(event: int, data: dict):
        LogHelper.info(f"全局事件触发: TRANSLATION_STOP_DONE, 全局状态={Base.WORK_STATUS}, 数据={data}")
        # 由于每个任务现在会单独处理完成事件，这里只处理没有被特定任务处理的情况
        active_tasks = []
        # 检查全局状态来区分自然完成和手动停止
        for task_id, task_info in list(translation_tasks.items()):
            if task_info.get("status") in ["translating", "stopping"]:
                active_tasks.append(task_id)
                # 如果仍有处于进行中的任务，更新它们的状态
                if Base.WORK_STATUS == Base.Status.STOPING:
                    LogHelper.info(f"全局处理: 将任务 {task_id} 标记为已停止")
                    translation_complete_callback(task_id, False, "翻译任务已被用户停止")
                else:
                    LogHelper.info(f"全局处理: 将任务 {task_id} 标记为已完成")
                    translation_complete_callback(task_id, True, "翻译任务已成功完成")

        if not active_tasks:
            LogHelper.info("全局事件: 没有活动的翻译任务需要处理")
        else:
            LogHelper.info(f"全局事件: 处理了 {len(active_tasks)} 个活动任务: {active_tasks}")

        # 重置全局状态为IDLE，除非有特殊情况
        if Base.WORK_STATUS != Base.Status.IDLE:
            old_status = Base.WORK_STATUS
            Base.WORK_STATUS = Base.Status.IDLE
            LogHelper.info(f"全局事件: 将全局状态从 {old_status} 重置为 {Base.WORK_STATUS}")

    # 注册翻译完成或停止事件
    translator.subscribe(Base.Event.TRANSLATION_STOP_DONE, on_translation_stop_done)
    LogHelper.info(f"已注册TRANSLATION_STOP_DONE事件处理器: {on_translation_stop_done}")

    # 注册翻译停止请求事件
    def on_translation_stop(event: int, data: dict):
        LogHelper.info(f"全局事件触发: TRANSLATION_STOP, 全局状态={Base.WORK_STATUS}, 数据={data}")
        # 设置全局状态为停止中
        old_status = Base.WORK_STATUS
        Base.WORK_STATUS = Base.Status.STOPING
        LogHelper.info(f"全局事件: 将全局状态从 {old_status} 设置为 {Base.WORK_STATUS}")

        # 当收到停止请求时，更新所有进行中任务的状态为"stopping"
        updated_tasks = []
        for task_id, task_info in translation_tasks.items():
            if task_info.get("status") == "translating":
                task_info.update({
                    "status": "stopping",
                    "message": "正在停止翻译任务..."
                })
                updated_tasks.append(task_id)

        if not updated_tasks:
            LogHelper.info("全局事件: 没有进行中的翻译任务需要停止")
        else:
            LogHelper.info(f"全局事件: 将 {len(updated_tasks)} 个任务标记为stopping: {updated_tasks}")

    # 注册事件
    translator.subscribe(Base.Event.TRANSLATION_STOP, on_translation_stop)
    LogHelper.info(f"已注册TRANSLATION_STOP事件处理器: {on_translation_stop}")

    # 添加对TRANSLATION_START事件的监听，以便诊断
    def on_translation_start(event: int, data: dict):
        LogHelper.info(f"全局事件触发: TRANSLATION_START, 全局状态={Base.WORK_STATUS}, 数据状态={data.get('status', 'unknown')}")
        LogHelper.info(f"TRANSLATION_START事件数据: {data}")

        # 检查translator的状态和属性
        if hasattr(translator, 'translating'):
            LogHelper.info(f"事件处理中检测到translator.translating = {translator.translating}")
        if hasattr(translator, 'status'):
            LogHelper.info(f"事件处理中检测到translator.status = {translator.status}")

        # 检查事件触发后是否有任何变化，用于诊断
        active_tasks = 0
        for task_id, task_info in translation_tasks.items():
            if task_info.get("status") in ["translating", "initializing"]:
                active_tasks += 1
                LogHelper.info(f"发现正在进行的任务 {task_id}: {task_info}")

        LogHelper.info(f"当前活动任务数: {active_tasks}")

        # 这里只记录事件，不做处理，以免干扰Translator类自身的处理

    translator.subscribe(Base.Event.TRANSLATION_START, on_translation_start)
    LogHelper.info(f"已注册TRANSLATION_START监听器(仅诊断): {on_translation_start}")

    # 查看所有已注册的事件处理器
    if hasattr(translator, '_handlers'):
        LogHelper.info("当前translator的所有事件处理器:")
        for event, handlers in translator._handlers.items():
            handlers_info = [f"{h.__name__ if hasattr(h, '__name__') else str(h)}" for h in handlers]
            LogHelper.info(f"  事件 {event}: {len(handlers)} 个处理器: {handlers_info}")

    LogHelper.info("全局事件处理器注册完成")

# 应用启动时连接事件
if not hasattr(connect_events, '_called'):
    connect_events()
    setattr(connect_events, '_called', True)
    LogHelper.info("首次连接全局事件，标记为已调用")
else:
    LogHelper.info("全局事件已经连接，跳过重复连接")

@app.get("/api/platforms", tags=["平台"])
def get_platforms():
    """
    获取所有平台配置
    """
    try:
        config = load_config()
        return PlatformsListResponse(
            platforms=config.get("platforms", []),
            activate_platform=config.get("activate_platform", 0)
        )
    except Exception as e:
        LogHelper.error(f"获取平台配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取平台配置失败: {str(e)}")

@app.put("/api/platforms/{platform_id}", tags=["平台"])
def update_platform(platform_id: int, platform_config: PlatformConfig):
    """
    更新指定平台配置
    """
    try:
        config = load_config()
        platforms = config.get("platforms", [])

        # 验证平台ID
        if platform_id != platform_config.id:
            raise HTTPException(status_code=400, detail="平台ID不匹配")

        # 查找并更新平台配置
        found = False
        for i, platform in enumerate(platforms):
            if platform.get("id") == platform_id:
                platforms[i] = platform_config.dict(exclude_unset=True)
                found = True
                break

        if not found:
            # 如果平台不存在，则添加新平台
            platforms.append(platform_config.dict(exclude_unset=True))

        # 更新配置
        config["platforms"] = platforms

        # 保存配置
        with open("resource/config.json", "w", encoding="utf-8-sig") as writer:
            import rapidjson as json
            json.dump(config, writer, indent=4)

        return {"status": "success", "message": f"平台 {platform_id} 已更新"}
    except HTTPException as e:
        raise e
    except Exception as e:
        LogHelper.error(f"更新平台配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新平台配置失败: {str(e)}")

@app.post("/api/platforms/activate/{platform_id}", tags=["平台"])
def activate_platform(platform_id: int):
    """
    激活指定平台
    """
    try:
        config = load_config()
        platforms = config.get("platforms", [])

        # 验证平台ID是否存在
        platform_exists = False
        for platform in platforms:
            if platform.get("id") == platform_id:
                platform_exists = True
                break

        if not platform_exists:
            raise HTTPException(status_code=404, detail=f"平台ID {platform_id} 不存在")

        # 更新激活平台
        config["activate_platform"] = platform_id

        # 保存配置
        with open("resource/config.json", "w", encoding="utf-8-sig") as writer:
            import rapidjson as json
            json.dump(config, writer, indent=4)

        return {"status": "success", "message": f"已激活平台 {platform_id}"}
    except HTTPException as e:
        raise e
    except Exception as e:
        LogHelper.error(f"激活平台失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"激活平台失败: {str(e)}")

# 启动服务器
if __name__ == "__main__":
    # 设置当前工作目录
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    sys.path.append(script_dir)

    # 初始化语言设置
    config = load_config()
    Localizer.set_app_language(config.get("app_language", Base.Language.ZH))

    # 打印启动日志
    LogHelper.info("LinguaGacha API 服务启动")

    # 启动API服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)