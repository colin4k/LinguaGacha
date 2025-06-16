#!/usr/bin/env python3
"""
LinguaGacha API启动脚本
基于CLI功能构建的API服务
"""

import os
import sys

# 设置工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

try:
    # 导入API模块
    from api import app
    from base.LogManager import LogManager
    from module.Localizer.Localizer import Localizer
    from module.Config import Config
    
    # 初始化配置
    config = Config().load()
    
    # 设置语言
    Localizer.set_app_language(config.app_language)
    
    # 创建必要的目录
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./resource", exist_ok=True)
    
    LogManager.get().info("LinguaGacha API 启动中...")
    
    # 启动API服务器
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
except ImportError as e:
    print(f"缺少依赖模块: {e}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"启动API服务失败: {e}")
    sys.exit(1)