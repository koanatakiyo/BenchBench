#!/usr/bin/env python3
"""
Quick sanity-check script for the Grok (x.ai) chat-completions endpoint.

Usage:
  python test_grok.py --api-key xai-... --message "Say hello"

If --api-key is omitted the script looks at the GROK_API_KEY or XAI_API_KEY
environment variables. The script prints the model response or a detailed
error message, which is useful for debugging Stage 2 Grok integration.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a simple prompt to Grok.")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Grok API key (falls back to GROK_API_KEY or XAI_API_KEY env vars)",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.x.ai/v1",
        help="Base URL for the Grok OpenAI-compatible API",
    )
    parser.add_argument(
        "--model",
        default="grok-4-latest",
        help="Model name to query",
    )
    parser.add_argument(
        "--message",
        default="Testing. Please respond with 'hello world'.",
        help="User message to send",
    )
    parser.add_argument(
        "--system",
        default="You are a friendly assistant used for connectivity tests.",
        help="Optional system prompt",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds",
    )
    return parser.parse_args()


def resolve_api_key(cli_key: Optional[str]) -> str:
    key = cli_key or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
    if not key:
        print(
            "Error: provide --api-key or set GROK_API_KEY/XAI_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(2)
    return key


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)

    payload = {
        "model": args.model,
        "stream": False,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.message},
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    url = f"{args.base_url.rstrip('/')}/chat/completions"

    with httpx.Client(timeout=args.timeout) as client:
        try:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            print(f"Grok returned HTTP {exc.response.status_code}: {exc.response.text}")
            sys.exit(1)
        except httpx.RequestError as exc:
            print(f"Grok request failed: {exc}")
            sys.exit(1)

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print(f"Unexpected Grok response: {data}")
        sys.exit(1)

    print("=== Grok Response ===")
    print(content.strip())


if __name__ == "__main__":
    main()

