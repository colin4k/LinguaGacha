import os
import sys
import time
import threading
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
from base.LogManager import LogManager  
from module.Config import Config
from module.Engine.Engine import Engine
from module.Engine.API.APITester import APITester
from module.Cache.CacheManager import CacheManager
from module.Cache.CacheItem import CacheItem
from module.Localizer.Localizer import Localizer

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
engine = Engine()
cache_manager = CacheManager(service=False)
translation_tasks = {}
config = Config()

# 初始化引擎
engine.run()

# 设置工作目录
sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0])))

# 挂载静态文件路径（如果存在）
if os.path.exists("static"):
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
def load_config() -> Config:
    return Config().load()

# 初始化目录
os.makedirs("./input", exist_ok=True)
os.makedirs("./output", exist_ok=True)

# 模型定义
class ApiConfig(BaseModel):
    # 应用基本配置
    app_language: Optional[str] = Field(default="ZH", description="应用语言(ZH/EN/JA)")
    theme: Optional[str] = Field(default="LIGHT", description="应用主题(LIGHT/DARK)")

    # 翻译配置
    source_language: Optional[str] = Field(default="JA", description="源语言")
    target_language: Optional[str] = Field(default="ZH", description="目标语言")
    input_folder: Optional[str] = Field(default="./input", description="输入文件夹路径")
    output_folder: Optional[str] = Field(default="./output", description="输出文件夹路径")
    traditional_chinese_enable: Optional[bool] = Field(default=False, description="是否启用繁体中文")
    token_threshold: Optional[int] = Field(default=384, description="翻译任务长度阈值")
    max_workers: Optional[int] = Field(default=0, description="最大工作线程数")
    rpm_threshold: Optional[int] = Field(default=0, description="RPM限制")
    request_timeout: Optional[int] = Field(default=120, description="请求超时时间(秒)")
    max_round: Optional[int] = Field(default=16, description="最大轮数")

    # 词汇表和替换配置
    auto_glossary_enable: Optional[bool] = Field(default=False, description="是否启用自动词汇表")
    mtool_optimizer_enable: Optional[bool] = Field(default=False, description="是否启用MTool优化器")
    glossary_enable: Optional[bool] = Field(default=True, description="是否启用词汇表")
    glossary_data: Optional[List[Dict[str, str]]] = Field(default=[], description="词汇表数据")
    pre_translation_replacement_enable: Optional[bool] = Field(default=True, description="是否启用翻译前替换")
    pre_translation_replacement_data: Optional[List[Dict[str, str]]] = Field(default=[], description="翻译前替换数据")
    post_translation_replacement_enable: Optional[bool] = Field(default=True, description="是否启用翻译后替换")
    post_translation_replacement_data: Optional[List[Dict[str, str]]] = Field(default=[], description="翻译后替换数据")

    # 专家设置
    expert_mode: Optional[bool] = Field(default=False, description="是否启用专家模式")
    preceding_lines_threshold: Optional[int] = Field(default=0, description="前置行数阈值")
    enable_preceding_on_local: Optional[bool] = Field(default=False, description="是否在本地启用前置")
    clean_ruby: Optional[bool] = Field(default=True, description="是否清理注音")
    
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
def update_config(api_config: ApiConfig):
    """
    更新翻译配置
    """
    try:
        # 获取当前配置
        current_config = load_config()

        # 更新配置
        for key, value in api_config.dict(exclude_unset=True).items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)

        # 保存配置
        current_config.save()

        return {"status": "success", "message": "配置已更新"}
    except Exception as e:
        LogManager.get().error(f"更新配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")

@app.get("/api/config", tags=["配置"])
def get_config():
    """
    获取当前翻译配置
    """
    try:
        config = load_config()
        return config.__dict__
    except Exception as e:
        LogManager.get().error(f"获取配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")

@app.post("/api/platform/test", tags=["平台"])
def test_platform(platform_config: PlatformConfig):
    """
    测试API平台连接
    """
    try:
        # 检查引擎状态
        if engine.get_status() != Engine.Status.IDLE:
            return PlatformResponse(
                status=False,
                message="引擎正在运行中，无法进行平台测试",
                platform_info={}
            )

        # 创建临时配置对象用于测试
        test_config = Config()
        if test_config.platforms is None:
            test_config.platforms = []
        
        # 添加测试平台配置
        platform_dict = platform_config.dict()
        test_config.platforms = [platform_dict]
        test_config.activate_platform = platform_config.id
        
        # 创建测试结果
        test_result = {"status": False, "message": "测试失败", "platform": platform_dict}
        
        def test_done_callback(event: str, data: dict):
            test_result["status"] = data.get("result", False)
            test_result["message"] = data.get("result_msg", "测试完成")
        
        # 订阅测试完成事件
        engine.api_test.subscribe(Base.Event.PLATFORM_TEST_DONE, test_done_callback)
        
        try:
            # 触发测试事件
            engine.api_test.emit(Base.Event.PLATFORM_TEST_START, {"id": platform_config.id})
            
            # 等待测试完成（最多30秒）
            for _ in range(300):  # 30秒 * 10 = 300次 * 0.1秒
                if test_result["status"] is not None:
                    break
                time.sleep(0.1)
        except Exception as test_error:
            test_result["message"] = f"测试过程中发生错误: {str(test_error)}"
        finally:
            # 取消订阅
            try:
                engine.api_test.unsubscribe(Base.Event.PLATFORM_TEST_DONE, test_done_callback)
            except:
                pass

        return PlatformResponse(
            status=test_result["status"],
            message=test_result["message"],
            platform_info=test_result.get("platform", {})
        )
    except Exception as e:
        LogManager.get().error(f"测试平台失败: {str(e)}")
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
        LogManager.get().error(f"上传文件失败: {str(e)}")
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
        LogManager.get().error(f"列出文件失败: {str(e)}")
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
        LogManager.get().error(f"下载文件失败: {str(e)}")
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
        LogManager.get().error(f"删除文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")

@app.post("/api/translation/start", tags=["翻译"])
def start_translation(background_tasks: BackgroundTasks):
    """
    开始翻译任务
    """
    try:
        # 检查引擎状态
        if engine.get_status() != Engine.Status.IDLE:
            raise HTTPException(status_code=400, detail="翻译引擎正在运行中，无法启动新的翻译任务")

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
    except HTTPException as e:
        raise e
    except Exception as e:
        LogManager.get().error(f"启动翻译任务失败: {str(e)}")
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
        LogManager.get().error(f"获取翻译状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取翻译状态失败: {str(e)}")

@app.post("/api/translation/stop/{task_id}", tags=["翻译"])
def stop_translation(task_id: str):
    """
    停止翻译任务
    """
    try:
        if task_id not in translation_tasks:
            raise HTTPException(status_code=404, detail=f"未找到ID为 {task_id} 的翻译任务")

        # 检查任务状态
        task_info = translation_tasks[task_id]
        if task_info.get("status") not in ["translating", "initializing"]:
            raise HTTPException(status_code=400, detail="只有进行中的翻译任务才能停止")

        # 设置停止标志
        translation_tasks[task_id]["status"] = "stopping"
        translation_tasks[task_id]["message"] = "正在停止翻译任务..."

        # 通知应用停止翻译
        engine.translator.emit(Base.Event.TRANSLATION_STOP, {})

        return {"status": "success", "message": "已发送停止翻译请求"}
    except HTTPException as e:
        raise e
    except Exception as e:
        LogManager.get().error(f"停止翻译任务失败: {str(e)}")
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
        engine.translator.emit(Base.Event.TRANSLATION_MANUAL_EXPORT, {})

        return {"status": "success", "message": "已发送导出翻译请求"}
    except HTTPException as e:
        raise e
    except Exception as e:
        LogManager.get().error(f"导出翻译结果失败: {str(e)}")
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
        cache_manager_temp = CacheManager(service=False)

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
        LogManager.get().error(f"启动继续翻译任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动继续翻译任务失败: {str(e)}")

# 翻译任务后台处理函数
def translation_task(task_id: str):
    try:
        LogManager.get().info(f"========== 开始翻译任务 {task_id} ==========")

        # 更新任务状态
        translation_tasks[task_id].update({
            "status": "translating",
            "message": "翻译任务进行中..."
        })

        # 创建进度回调
        def progress_callback(event: str, data: dict):
            LogManager.get().info(f"收到翻译进度更新: event={event}, 任务ID={task_id}")
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
                        LogManager.get().info(f"更新翻译进度: {done}/{total}, 进度={progress:.2f}")

        # 创建翻译完成回调
        def complete_callback(event: str, data: dict):
            LogManager.get().info(f"收到翻译完成事件: TRANSLATION_DONE, 任务ID={task_id}")
            if task_id in translation_tasks:
                # 检查引擎状态来区分自然完成和手动停止
                if engine.get_status() == Engine.Status.STOPPING:
                    translation_tasks[task_id].update({
                        "status": "stopped",
                        "message": "翻译任务已被用户停止"
                    })
                else:
                    translation_tasks[task_id].update({
                        "status": "completed",
                        "progress": 1.0,
                        "message": "翻译任务已成功完成"
                    })

            # 取消订阅，避免内存泄漏
            engine.translator.unsubscribe(Base.Event.TRANSLATION_DONE, complete_callback)
            engine.translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)

        # 注册事件回调
        engine.translator.subscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
        engine.translator.subscribe(Base.Event.TRANSLATION_DONE, complete_callback)

        # 加载配置
        current_config = load_config()

        # 检查输入文件夹中是否有文件
        files = []
        for root, _, filenames in os.walk(current_config.input_folder):
            for filename in filenames:
                files.append(os.path.join(root, filename))

        if not files:
            LogManager.get().error("输入文件夹中没有找到文件，无法启动翻译")
            translation_tasks[task_id].update({
                "status": "failed",
                "message": "输入文件夹中没有找到文件，无法启动翻译"
            })
            # 取消订阅
            engine.translator.unsubscribe(Base.Event.TRANSLATION_DONE, complete_callback)
            engine.translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
            return

        LogManager.get().info(f"找到输入文件 {len(files)} 个")

        # 触发翻译开始事件，使用CLI架构
        engine.translator.emit(Base.Event.TRANSLATION_START, {
            "config": current_config,
            "status": Base.TranslationStatus.UNTRANSLATED,
        })

        LogManager.get().info("翻译任务已启动")

    except Exception as e:
        LogManager.get().error(f"翻译任务启动失败: {str(e)}")
        if task_id in translation_tasks:
            translation_tasks[task_id].update({
                "status": "failed",
                "message": f"翻译任务启动失败: {str(e)}"
            })

        # 确保取消订阅
        try:
            engine.translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
            engine.translator.unsubscribe(Base.Event.TRANSLATION_DONE, complete_callback)
        except Exception as ex:
            LogManager.get().error(f"取消订阅时出错: {str(ex)}")

# 继续翻译任务后台处理函数
def continue_translation_task(task_id: str):
    try:
        LogManager.get().info(f"========== 开始继续翻译任务 {task_id} ==========")

        # 更新任务状态
        translation_tasks[task_id].update({
            "status": "translating",
            "message": "继续翻译任务进行中..."
        })

        # 创建进度回调
        def progress_callback(event: str, data: dict):
            LogManager.get().info(f"收到继续翻译进度更新: event={event}, 任务ID={task_id}")
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
                        LogManager.get().info(f"更新继续翻译进度: {done}/{total}, 进度={progress:.2f}")

        # 创建翻译完成回调
        def complete_callback(event: str, data: dict):
            LogManager.get().info(f"收到继续翻译完成事件: TRANSLATION_DONE, 任务ID={task_id}")
            if task_id in translation_tasks:
                # 检查引擎状态来区分自然完成和手动停止
                if engine.get_status() == Engine.Status.STOPPING:
                    translation_tasks[task_id].update({
                        "status": "stopped",
                        "message": "继续翻译任务已被用户停止"
                    })
                else:
                    translation_tasks[task_id].update({
                        "status": "completed",
                        "progress": 1.0,
                        "message": "继续翻译任务已成功完成"
                    })

            # 取消订阅，避免内存泄漏
            engine.translator.unsubscribe(Base.Event.TRANSLATION_DONE, complete_callback)
            engine.translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)

        # 注册事件回调
        engine.translator.subscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
        engine.translator.subscribe(Base.Event.TRANSLATION_DONE, complete_callback)

        # 检查是否有保存的翻译项目
        current_config = load_config()
        
        # 检查是否有可继续的翻译项目
        temp_cache_manager = CacheManager(service=False)
        if not temp_cache_manager.load_project_from_file(current_config.output_folder):
            LogManager.get().error("没有找到可继续的翻译项目")
            translation_tasks[task_id].update({
                "status": "failed",
                "message": "没有找到可继续的翻译项目"
            })
            # 取消订阅
            engine.translator.unsubscribe(Base.Event.TRANSLATION_DONE, complete_callback)
            engine.translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
            return

        project_status = temp_cache_manager.get_project().get_status()
        LogManager.get().info(f"项目状态: {project_status}")

        if project_status != Base.TranslationStatus.TRANSLATING:
            LogManager.get().error("没有处于翻译中状态的项目可继续")
            translation_tasks[task_id].update({
                "status": "failed",
                "message": "没有处于翻译中状态的项目可继续"
            })
            # 取消订阅
            engine.translator.unsubscribe(Base.Event.TRANSLATION_DONE, complete_callback)
            engine.translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
            return

        # 触发继续翻译开始事件，使用CLI架构
        engine.translator.emit(Base.Event.TRANSLATION_START, {
            "config": current_config,
            "status": Base.TranslationStatus.TRANSLATING,  # 使用TRANSLATING表示继续翻译
        })

        LogManager.get().info("继续翻译任务已启动")

    except Exception as e:
        LogManager.get().error(f"继续翻译任务启动失败: {str(e)}")
        if task_id in translation_tasks:
            translation_tasks[task_id].update({
                "status": "failed",
                "message": f"继续翻译任务启动失败: {str(e)}"
            })

        # 确保取消订阅
        try:
            engine.translator.unsubscribe(Base.Event.TRANSLATION_UPDATE, progress_callback)
            engine.translator.unsubscribe(Base.Event.TRANSLATION_DONE, complete_callback)
        except Exception as ex:
            LogManager.get().error(f"取消订阅时出错: {str(ex)}")


@app.get("/api/platforms", tags=["平台"])
def get_platforms():
    """
    获取所有平台配置
    """
    try:
        config = load_config()
        return PlatformsListResponse(
            platforms=config.platforms if config.platforms else [],
            activate_platform=config.activate_platform
        )
    except Exception as e:
        LogManager.get().error(f"获取平台配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取平台配置失败: {str(e)}")

@app.put("/api/platforms/{platform_id}", tags=["平台"])
def update_platform(platform_id: int, platform_config: PlatformConfig):
    """
    更新指定平台配置
    """
    try:
        config = load_config()
        
        # 验证平台ID
        if platform_id != platform_config.id:
            raise HTTPException(status_code=400, detail="平台ID不匹配")

        # 初始化platforms列表（如果为None）
        if config.platforms is None:
            config.platforms = []

        # 查找并更新平台配置
        platform_dict = platform_config.dict(exclude_unset=True)
        found = False
        for i, platform in enumerate(config.platforms):
            if platform.get("id") == platform_id:
                config.platforms[i] = platform_dict
                found = True
                break

        if not found:
            # 如果平台不存在，则添加新平台
            config.platforms.append(platform_dict)

        # 保存配置
        config.save()

        return {"status": "success", "message": f"平台 {platform_id} 已更新"}
    except HTTPException as e:
        raise e
    except Exception as e:
        LogManager.get().error(f"更新平台配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新平台配置失败: {str(e)}")

@app.post("/api/platforms/activate/{platform_id}", tags=["平台"])
def activate_platform(platform_id: int):
    """
    激活指定平台
    """
    try:
        config = load_config()

        # 验证平台ID是否存在
        platform_exists = False
        if config.platforms:
            for platform in config.platforms:
                if platform.get("id") == platform_id:
                    platform_exists = True
                    break

        if not platform_exists:
            raise HTTPException(status_code=404, detail=f"平台ID {platform_id} 不存在")

        # 更新激活平台
        config.activate_platform = platform_id

        # 保存配置
        config.save()

        return {"status": "success", "message": f"已激活平台 {platform_id}"}
    except HTTPException as e:
        raise e
    except Exception as e:
        LogManager.get().error(f"激活平台失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"激活平台失败: {str(e)}")

# 启动服务器
if __name__ == "__main__":
    # 设置当前工作目录
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    sys.path.append(script_dir)

    # 初始化语言设置
    startup_config = load_config()
    Localizer.set_app_language(startup_config.app_language)

    # 打印启动日志
    LogManager.get().info("LinguaGacha API 服务启动")

    # 启动API服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)