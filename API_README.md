# LinguaGacha API 使用文档

## 概述

LinguaGacha API 是基于现有CLI功能构建的RESTful API服务，提供完整的翻译功能。API保持与CLI相同的翻译逻辑和配置管理，确保功能一致性。

## 启动API服务

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python start_api.py
```

或者直接运行：

```bash
python api.py
```

服务默认监听 `http://localhost:8000`

### 3. 查看API文档

启动服务后，访问：
- Swagger UI: `http://localhost:8000/docs`
- API详细信息: `http://localhost:8000/`

## API端点

### 状态检查

- `GET /` - 获取API服务状态

### 配置管理

- `GET /api/config` - 获取当前配置
- `POST /api/config` - 更新配置

### 平台管理

- `GET /api/platforms` - 获取所有平台配置
- `PUT /api/platforms/{platform_id}` - 更新平台配置
- `POST /api/platforms/activate/{platform_id}` - 激活指定平台
- `POST /api/platform/test` - 测试平台连接

### 文件管理

- `GET /api/files/list` - 列出输入/输出文件
- `POST /api/files/upload` - 上传文件到input文件夹
- `GET /api/files/download/{file_path}` - 下载文件
- `DELETE /api/files/delete/{file_path}` - 删除文件

### 翻译任务

- `POST /api/translation/start` - 开始新的翻译任务
- `GET /api/translation/status/{task_id}` - 获取翻译任务状态
- `POST /api/translation/stop/{task_id}` - 停止翻译任务
- `POST /api/translation/continue` - 继续上次的翻译任务
- `POST /api/translation/export/{task_id}` - 手动导出翻译结果

## 使用示例

### 1. 基本工作流程

```python
import requests

API_BASE = "http://localhost:8000"

# 1. 检查服务状态
response = requests.get(f"{API_BASE}/")
print(response.json())

# 2. 获取当前配置
config = requests.get(f"{API_BASE}/api/config").json()
print(f"当前源语言: {config['source_language']}")

# 3. 更新配置
new_config = {
    "source_language": "JA",
    "target_language": "ZH",
    "max_workers": 4
}
requests.post(f"{API_BASE}/api/config", json=new_config)

# 4. 上传文件
with open("text.txt", "rb") as f:
    files = {"file": f}
    requests.post(f"{API_BASE}/api/files/upload", files=files)

# 5. 开始翻译
response = requests.post(f"{API_BASE}/api/translation/start")
task_id = response.json()["task_id"]

# 6. 监控翻译进度
import time
while True:
    status = requests.get(f"{API_BASE}/api/translation/status/{task_id}").json()
    print(f"进度: {status['progress']:.2%} - {status['message']}")
    
    if status["status"] in ["completed", "failed", "stopped"]:
        break
    time.sleep(5)

# 7. 下载结果文件
files = requests.get(f"{API_BASE}/api/files/list").json()
for file_path in files["output_files"]:
    response = requests.get(f"{API_BASE}/api/files/download/{file_path}")
    with open(f"translated_{file_path.split('/')[-1]}", "wb") as f:
        f.write(response.content)
```

### 2. 平台配置示例

```python
# 配置OpenAI平台
platform_config = {
    "id": 1,
    "name": "OpenAI",
    "api_url": "https://api.openai.com/v1",
    "api_key": ["your-api-key-here"],
    "api_format": "OpenAI",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "top_p": 0.9
}

# 添加/更新平台
requests.put(f"{API_BASE}/api/platforms/1", json=platform_config)

# 激活平台
requests.post(f"{API_BASE}/api/platforms/activate/1")

# 测试平台连接
test_result = requests.post(f"{API_BASE}/api/platform/test", json=platform_config)
print(f"测试结果: {test_result.json()}")
```

## 数据模型

### ApiConfig (配置)

主要配置字段：
- `source_language`: 源语言 (默认: "JA")
- `target_language`: 目标语言 (默认: "ZH") 
- `input_folder`: 输入文件夹 (默认: "./input")
- `output_folder`: 输出文件夹 (默认: "./output")
- `max_workers`: 最大工作线程数 (默认: 0)
- `request_timeout`: 请求超时时间 (默认: 120秒)
- `max_round`: 最大轮数 (默认: 16)

### PlatformConfig (平台配置)

主要字段：
- `id`: 平台ID
- `name`: 平台名称
- `api_url`: API地址
- `api_key`: API密钥列表
- `api_format`: API格式 (OpenAI/Google/Anthropic/SakuraLLM)
- `model`: 模型名称
- `temperature`: 温度参数
- `top_p`: top_p参数

### TranslationStatusResponse (翻译状态)

- `task_id`: 任务ID
- `status`: 状态 (initializing/translating/completed/failed/stopped)
- `progress`: 进度 (0.0-1.0)
- `total_items`: 总项目数
- `translated_items`: 已翻译项目数
- `message`: 状态消息

## 注意事项

1. **引擎状态**: 同一时间只能运行一个翻译任务
2. **文件安全**: 文件操作限制在项目目录内，防止路径遍历攻击
3. **事件系统**: API使用与CLI相同的事件系统，确保功能一致性
4. **配置持久化**: 配置自动保存到 `resource/config.json`
5. **日志记录**: 使用项目的LogManager进行日志记录

## 测试

运行测试脚本：

```bash
python test_api.py
```

确保API服务正在运行后执行测试。

## 故障排除

1. **导入错误**: 确保安装了所有依赖 `pip install -r requirements.txt`
2. **端口占用**: 修改 `start_api.py` 中的端口号
3. **权限问题**: 确保有读写 `input`、`output`、`resource` 目录的权限
4. **翻译失败**: 检查平台配置和API密钥是否正确