import requests
import time
import sys

def test_endpoints():
    base_url = "http://localhost:8000"
    
    print("测试vLLM服务状态...")
    print("-" * 50)
    
    # 测试1: 健康检查
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        print(f"1. /health → Status: {resp.status_code}, Response: {resp.text}")
    except Exception as e:
        print(f"1. /health → 失败: {e}")
    
    # 测试2: 模型列表
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        print(f"2. /v1/models → Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"   模型列表: {resp.json()}")
    except Exception as e:
        print(f"2. /v1/models → 失败: {e}")
    
    # 测试3: 聊天接口
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "llama-70b",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            },
            timeout=10
        )
        print(f"3. /v1/chat/completions → Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"   响应: {resp.json()}")
        else:
            print(f"   错误: {resp.text}")
    except Exception as e:
        print(f"3. /v1/chat/completions → 失败: {e}")
    
    print("-" * 50)

if __name__ == "__main__":
    # 先等5秒让服务完全启动
    time.sleep(5)
    test_endpoints()