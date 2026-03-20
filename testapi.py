#!/usr/bin/python
# -*- coding: utf-8 -*-
# testapi.py - AI 模型 API 通用测试工具
# 支持测试: OpenAI, Google Gemini, xAI Grok, Anthropic Claude, 国内模型(qwen/deepseek/kimi/glm/minmax)

import os
import sys
import json
import time
import argparse
from datetime import datetime

try:
    import requests
except ImportError:
    print("请先安装 requests 库: pip install requests")
    sys.exit(1)

API_KEY = "<API_KEY>"
BASE_URL = "<URL>"
MAX_TOKENS = 100


class APITester:
    """AI 模型 API 通用测试器"""

    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url or BASE_URL
        self.api_key = api_key or API_KEY
        self.results = []
        self.test_message = "你好"
        self.max_tokens = MAX_TOKENS
        self.auto_retry = True

        self.model_configs = {
            "openai": {
                "name": "OpenAI (ChatGPT)",
                "endpoints": [
                    "/v1/chat/completions",
                    "/v1/completions",
                    "/v1/responses"
                ],
                "payload_builder": lambda msg, model: {
                    "model": model or "gpt-4o-mini",
                    "messages": [{"role": "user", "content": msg}],
                    "max_tokens": self.max_tokens
                },
                "response_parser": lambda data: data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "auth_header": lambda key: {"Authorization": f"Bearer {key}"}
            },
            "gemini": {
                "name": "Google Gemini",
                "endpoints": [
                    "/v1beta/models/gemini-2.0-flash:generateContent",
                    "/v1/models/gemini-2.0-flash:generateContent",
                    "/v1beta/models/gemini-pro:generateContent"
                ],
                "payload_builder": lambda msg, model: {
                    "contents": [{"parts": [{"text": msg}]}],
                    "generationConfig": {"maxOutputTokens": self.max_tokens}
                },
                "response_parser": lambda data: data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", ""),
                "auth_header": lambda key: {"Authorization": f"Bearer {key}"}
            },
            "grok": {
                "name": "xAI Grok",
                "endpoints": [
                    "/v1/chat/completions",
                    "/v1/completions"
                ],
                "payload_builder": lambda msg, model: {
                    "model": model or "grok-2-1212",
                    "messages": [{"role": "user", "content": msg}],
                    "max_tokens": self.max_tokens
                },
                "response_parser": lambda data: data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "auth_header": lambda key: {"Authorization": f"Bearer {key}"}
            },
            "claude": {
                "name": "Anthropic Claude",
                "endpoints": [
                    "/v1/messages",
                    "/v1/messages/claude-3-5-sonnet-20241022"
                ],
                "payload_builder": lambda msg, model: {
                    "model": model or "claude-sonnet-4-20250514",
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": msg}]
                },
                "response_parser": lambda data: data.get("content", [{}])[0].get("text", ""),
                "auth_header": lambda key: {"x-api-key": key, "anthropic-version": "2023-06-01"}
            },
            "qwen": {
                "name": "阿里 Qwen",
                "endpoints": [
                    "/v1/chat/completions",
                    "/qwen/chat/completions"
                ],
                "payload_builder": lambda msg, model: {
                    "model": model or "qwen-turbo",
                    "messages": [{"role": "user", "content": msg}],
                    "max_tokens": self.max_tokens
                },
                "response_parser": lambda data: data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "auth_header": lambda key: {"Authorization": f"Bearer {key}"}
            },
            "deepseek": {
                "name": "DeepSeek",
                "endpoints": [
                    "/v1/chat/completions",
                    "/deepseek/chat/completions"
                ],
                "payload_builder": lambda msg, model: {
                    "model": model or "deepseek-chat",
                    "messages": [{"role": "user", "content": msg}],
                    "max_tokens": self.max_tokens
                },
                "response_parser": lambda data: data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "auth_header": lambda key: {"Authorization": f"Bearer {key}"}
            },
            "kimi": {
                "name": "月之暗面 Kimi",
                "endpoints": [
                    "/v1/chat/completions",
                    "/kimi/chat/completions"
                ],
                "payload_builder": lambda msg, model: {
                    "model": model or "kimi-flash",
                    "messages": [{"role": "user", "content": msg}],
                    "max_tokens": self.max_tokens
                },
                "response_parser": lambda data: data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "auth_header": lambda key: {"Authorization": f"Bearer {key}"}
            },
            "glm": {
                "name": "智谱 GLM",
                "endpoints": [
                    "/v1/chat/completions",
                    "/glm/chat/completions"
                ],
                "payload_builder": lambda msg, model: {
                    "model": model or "glm-4-flash",
                    "messages": [{"role": "user", "content": msg}],
                    "max_tokens": self.max_tokens
                },
                "response_parser": lambda data: data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "auth_header": lambda key: {"Authorization": f"Bearer {key}"}
            },
            "minmax": {
                "name": "Minimax",
                "endpoints": [
                    "/v1/chat/completions",
                    "/minimax/chat/completions"
                ],
                "payload_builder": lambda msg, model: {
                    "model": model or "abab6.5s-chat",
                    "messages": [{"role": "user", "content": msg}],
                    "max_tokens": self.max_tokens
                },
                "response_parser": lambda data: data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "auth_header": lambda key: {"Authorization": f"Bearer {key}"}
            },
            "yi": {
                "name": "零一万物 Yi",
                "endpoints": [
                    "/v1/chat/completions",
                    "/yi/chat/completions"
                ],
                "payload_builder": lambda msg, model: {
                    "model": model or "yi-medium",
                    "messages": [{"role": "user", "content": msg}],
                    "max_tokens": self.max_tokens
                },
                "response_parser": lambda data: data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "auth_header": lambda key: {"Authorization": f"Bearer {key}"}
            }
        }

    def print_header(self, mode="自动测试"):
        """打印测试头部信息"""
        print("=" * 70)
        print("       AI 模型 API 通用测试工具")
        print("=" * 70)
        print(f"模式: {mode}")
        print(f"API 地址: {self.base_url}")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试消息: {self.test_message}")
        print(f"最大Token: {self.max_tokens}")
        print("=" * 70)

    def print_result(self, name, status, message, response_time=0, endpoint=""):
        """打印测试结果"""
        status_icon = "✅" if status == "可用" else "❌" if status == "不可用" else "⚠️"
        print(f"{status_icon} {name}: {status}")
        if message:
            print(f"   └─ {message}")
        if response_time > 0:
            print(f"   └─ 响应时间: {response_time:.2f}s")
        if endpoint:
            print(f"   └─ 端点: {endpoint}")

        self.results.append({
            "name": name,
            "status": status,
            "message": message,
            "response_time": response_time,
            "endpoint": endpoint
        })

    def analyze_error(self, response_text, status_code):
        """分析错误原因并返回诊断信息"""
        error_analysis = {
            "status_code": status_code,
            "error_type": "unknown",
            "suggestion": "请检查API配置"
        }

        try:
            error_data = json.loads(response_text)
            error_msg = error_data.get("error", {}).get("message", "") or str(error_data)

            error_analysis["error_message"] = error_msg

            if status_code == 401:
                error_analysis["error_type"] = "认证失败"
                error_analysis["suggestion"] = "API Key无效或已过期"
            elif status_code == 403:
                error_analysis["error_type"] = "权限不足"
                error_analysis["suggestion"] = "API Key没有访问权限"
            elif status_code == 404:
                error_analysis["error_type"] = "端点不存在"
                error_analysis["suggestion"] = "尝试其他端点路径"
            elif status_code == 429:
                error_analysis["error_type"] = "请求过于频繁"
                error_analysis["suggestion"] = "等待后重试或降低请求频率"
            elif status_code == 500:
                error_analysis["error_type"] = "服务器内部错误"
                error_analysis["suggestion"] = "服务端问题，稍后重试"
            elif status_code == 503:
                error_analysis["error_type"] = "服务不可用"
                error_analysis["suggestion"] = "该模型当前无配额或通道"
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                error_analysis["error_type"] = "模型不存在"
                error_analysis["suggestion"] = "请检查模型名称是否正确"
            elif "api key" in error_msg.lower():
                error_analysis["error_type"] = "API Key错误"
                error_analysis["suggestion"] = "请检查API Key配置"
            else:
                error_analysis["error_type"] = "未知错误"
                error_analysis["suggestion"] = f"错误信息: {error_msg[:100]}"

        except:
            error_analysis["error_message"] = response_text[:200] if response_text else "未知错误"
            error_analysis["error_type"] = "解析失败"
            error_analysis["suggestion"] = "请检查API响应格式"

        return error_analysis

    def test_model(self, model_key, custom_model=None):
        """测试单个模型"""
        if model_key not in self.model_configs:
            self.print_result(model_key, "未配置", "不支持的模型类型")
            return None

        config = self.model_configs[model_key]
        model_name = custom_model or config.get("default_model", "")

        for endpoint in config["endpoints"]:
            try:
                start_time = time.time()

                url = f"{self.base_url}{endpoint}"
                payload = config["payload_builder"](self.test_message, custom_model)
                headers = {
                    **config["auth_header"](self.api_key),
                    "Content-Type": "application/json"
                }

                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response_time = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    reply = config["response_parser"](data)
                    self.print_result(
                        config["name"],
                        "可用",
                        f"回复: {reply[:50]}...",
                        response_time,
                        endpoint
                    )
                    return True
                else:
                    error_analysis = self.analyze_error(response.text, response.status_code)

                    if error_analysis["error_type"] == "端点不存在" and self.auto_retry:
                        continue

                    self.print_result(
                        config["name"],
                        "不可用",
                        f"HTTP {response.status_code} - {error_analysis['error_type']}: {error_analysis['suggestion']}",
                        response_time,
                        endpoint
                    )
                    return False

            except requests.exceptions.Timeout:
                self.print_result(config["name"], "不可用", "请求超时", 0, endpoint)
                return False
            except requests.exceptions.ConnectionError:
                if endpoint == config["endpoints"][-1]:
                    self.print_result(config["name"], "不可用", "连接失败，请检查网络或URL", 0, endpoint)
                continue
            except Exception as e:
                if endpoint == config["endpoints"][-1]:
                    self.print_result(config["name"], "不可用", f"异常: {str(e)}", 0, endpoint)
                continue

        self.print_result(config["name"], "不可用", "所有端点均失败", 0, endpoint)
        return False

    def test_all_auto(self):
        """自动测试所有模型"""
        self.print_header("自动测试")
        print("\n开始自动测试所有模型...\n")

        for model_key in self.model_configs.keys():
            self.test_model(model_key)

        self.print_summary()

    def test_specified(self, model_list):
        """指定测试特定模型"""
        self.print_header("指定测试")
        print(f"\n测试指定模型: {', '.join(model_list)}\n")

        for model_key in model_list:
            if model_key in self.model_configs:
                self.test_model(model_key)
            else:
                self.print_result(model_key, "未配置", "不支持的模型类型")

        self.print_summary()

    def test_custom(self, model_name, endpoint_type="chat"):
        """手动自定义测试"""
        self.print_header("手动测试")
        print(f"\n模型: {model_name}")
        print(f"访问方式: {endpoint_type}\n")

        endpoint_map = {
            "chat": "/v1/chat/completions",
            "completion": "/v1/completions",
            "responses": "/v1/responses",
            "message": "/v1/messages"
        }

        endpoint = endpoint_map.get(endpoint_type, f"/v1/{endpoint_type}/completions")

        try:
            start_time = time.time()

            if endpoint_type in ["chat", "completion"]:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": self.test_message}],
                    "max_tokens": self.max_tokens
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            elif endpoint_type == "message":
                payload = {
                    "model": model_name,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": self.test_message}]
                }
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
            else:
                payload = {
                    "model": model_name,
                    "prompt": self.test_message,
                    "max_tokens": self.max_tokens
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "") or \
                        data.get("content", [{}])[0].get("text", "") or \
                        data.get("text", "")
                self.print_result(model_name, "可用", f"回复: {reply[:100]}...", response_time, endpoint)
            else:
                error_analysis = self.analyze_error(response.text, response.status_code)
                self.print_result(
                    model_name,
                    "不可用",
                    f"HTTP {response.status_code} - {error_analysis['error_type']}: {error_analysis['suggestion']}",
                    response_time,
                    endpoint
                )

        except requests.exceptions.Timeout:
            self.print_result(model_name, "不可用", "请求超时", 0, endpoint)
        except Exception as e:
            self.print_result(model_name, "不可用", f"异常: {str(e)}", 0, endpoint)

        self.print_summary()

    def print_summary(self):
        """打印测试汇总"""
        print("\n" + "=" * 70)
        print("测试汇总")
        print("=" * 70)

        available = sum(1 for r in self.results if r["status"] == "可用")
        unavailable = sum(1 for r in self.results if r["status"] == "不可用")
        not_configured = sum(1 for r in self.results if r["status"] == "未配置")

        print(f"可用:      {available}")
        print(f"不可用:    {unavailable}")
        print(f"未配置:    {not_configured}")
        print(f"总计:      {len(self.results)}")

        if available > 0:
            print(f"\n可用的 API:")
            for r in self.results:
                if r["status"] == "可用":
                    print(f"  - {r['name']} ({r['response_time']:.2f}s)")

        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="AI 模型 API 通用测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  自动测试所有模型:
    python testapi.py --auto

  指定测试特定模型:
    python testapi.py --specify openai gemini qwen

  手动自定义测试:
    python testapi.py --custom gpt-4o-mini --type chat

  指定自定义API地址:
    python testapi.py --auto --url https://api.custom.com --key your-api-key

支持测试的模型类型:
  openai, gemini, grok, claude
  qwen, deepseek, kimi, glm, minimax, yi

支持的手动测试访问方式:
  chat (聊天), completion (补全), responses (响应), message (消息)
        """
    )

    parser.add_argument("--auto", action="store_true", help="自动测试所有支持的模型")
    parser.add_argument("--specify", nargs="+", help="指定测试的模型列表")
    parser.add_argument("--custom", type=str, help="手动自定义测试: 模型名称")
    parser.add_argument("--type", type=str, default="chat", choices=["chat", "completion", "responses", "message"],
                        help="手动测试的访问方式 (默认: chat)")
    parser.add_argument("--url", type=str, default=BASE_URL, help=f"API base URL (默认: {BASE_URL})")
    parser.add_argument("--key", type=str, default=API_KEY, help="API Key")
    parser.add_argument("--msg", type=str, default="你好", help="测试消息 (默认: 你好)")
    parser.add_argument("--tokens", type=int, default=MAX_TOKENS, help=f"最大token数 (默认: {MAX_TOKENS})")

    args = parser.parse_args()

    if args.tokens > 100:
        print(f"警告: token数已限制为100，当前设置为{args.tokens}，将使用100")
        args.tokens = 100

    tester = APITester(base_url=args.url, api_key=args.key)
    tester.test_message = args.msg
    tester.max_tokens = args.tokens

    if args.custom:
        tester.test_custom(args.custom, args.type)
    elif args.specify:
        tester.test_specified(args.specify)
    elif args.auto:
        tester.test_all_auto()
    else:
        print("请指定测试模式: --auto, --specify <模型列表>, 或 --custom <模型名>")
        print("使用 --help 查看更多帮助信息")


if __name__ == "__main__":
    main()
