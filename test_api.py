#!/usr/bin/env python3
"""
LinguaGacha API测试脚本
"""

import json
import requests
import time

API_BASE = "http://localhost:8000"

def test_api_basic():
    """测试API基本功能"""
    print("=== 测试API基本功能 ===")
    
    try:
        # 测试根路径
        response = requests.get(f"{API_BASE}/")
        print(f"根路径响应: {response.status_code}")
        if response.status_code == 200:
            print(f"响应内容: {response.json()}")
    except Exception as e:
        print(f"无法连接API服务: {e}")
        return False
    
    # 测试获取配置
    try:
        response = requests.get(f"{API_BASE}/api/config")
        print(f"获取配置响应: {response.status_code}")
        if response.status_code == 200:
            config = response.json()
            print(f"当前配置keys: {list(config.keys())}")
    except Exception as e:
        print(f"获取配置失败: {e}")
    
    # 测试文件列表
    try:
        response = requests.get(f"{API_BASE}/api/files/list")
        print(f"文件列表响应: {response.status_code}")
        if response.status_code == 200:
            files = response.json()
            print(f"输入文件数: {len(files.get('input_files', []))}")
            print(f"输出文件数: {len(files.get('output_files', []))}")
    except Exception as e:
        print(f"获取文件列表失败: {e}")
    
    # 测试平台列表
    try:
        response = requests.get(f"{API_BASE}/api/platforms")
        print(f"平台列表响应: {response.status_code}")
        if response.status_code == 200:
            platforms = response.json()
            print(f"平台数量: {len(platforms.get('platforms', []))}")
    except Exception as e:
        print(f"获取平台列表失败: {e}")
    
    return True

def test_api_config():
    """测试配置API"""
    print("\n=== 测试配置API ===")
    
    try:
        # 更新配置
        new_config = {
            "source_language": "JA",
            "target_language": "ZH",
            "max_workers": 4
        }
        
        response = requests.post(f"{API_BASE}/api/config", json=new_config)
        print(f"更新配置响应: {response.status_code}")
        
        if response.status_code == 200:
            print("配置更新成功")
            
            # 验证配置是否更新
            response = requests.get(f"{API_BASE}/api/config")
            if response.status_code == 200:
                config = response.json()
                print(f"验证更新 - source_language: {config.get('source_language')}")
                print(f"验证更新 - target_language: {config.get('target_language')}")
                print(f"验证更新 - max_workers: {config.get('max_workers')}")
        else:
            print(f"配置更新失败: {response.text}")
            
    except Exception as e:
        print(f"配置测试失败: {e}")

if __name__ == "__main__":
    print("LinguaGacha API 测试")
    print("请确保API服务正在运行 (python start_api.py)")
    print("等待3秒后开始测试...")
    time.sleep(3)
    
    if test_api_basic():
        test_api_config()
    
    print("\n测试完成")