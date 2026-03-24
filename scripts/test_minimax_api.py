#!/usr/bin/env python3
"""Minimal MiniMax API connectivity/auth test for remote servers."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_MODEL = "MiniMax-M2.5"


def mask_key(api_key: str) -> str:
    """Mask an API key for terminal output."""
    if len(api_key) <= 10:
        return "*" * len(api_key)
    return f"{api_key[:6]}...{api_key[-4:]}"


def build_payload(args: argparse.Namespace) -> dict:
    """Build a minimal OpenAI-compatible MiniMax chat request."""
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.user_prompt},
        ],
    }
    if args.reasoning_split:
        payload["reasoning_split"] = True
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    return payload


def extract_text(response_json: dict) -> tuple[str | None, str | None]:
    """Extract visible content and reasoning content from a chat completion."""
    choices = response_json.get("choices") or []
    if not choices:
        return None, None

    message = choices[0].get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")

    if isinstance(content, str):
        content_text = content.strip() or None
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text") or item.get("value") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        content_text = "\n".join(parts).strip() or None
    else:
        content_text = None

    if isinstance(reasoning, str):
        reasoning_text = reasoning.strip() or None
    else:
        reasoning_text = None

    return content_text, reasoning_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Test MiniMax chat/completions authentication.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MINIMAX_BASE_URL", DEFAULT_BASE_URL),
        help=(
            "MiniMax API base URL. Global default: https://api.minimax.io/v1 . "
            "China console often uses: https://api.minimaxi.com/v1"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MINIMAX_API_KEY"),
        help="MiniMax API key. Defaults to MINIMAX_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MINIMAX_MODEL", DEFAULT_MODEL),
        help=f"Model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a concise assistant. Reply briefly.",
        help="System prompt for the test request.",
    )
    parser.add_argument(
        "--user-prompt",
        default='Reply with exactly this JSON: {"ok": true, "provider": "minimax"}',
        help="User prompt for the test request.",
    )
    parser.add_argument(
        "--reasoning-split",
        action="store_true",
        help="Include reasoning_split=true in the request body.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature to include in the request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Optional max_tokens to include in the request.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print the full JSON response body.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: missing API key. Set MINIMAX_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    payload = build_payload(args)

    print("MiniMax API test")
    print(f"Endpoint: {endpoint}")
    print(f"Model: {args.model}")
    print(f"API key: {mask_key(args.api_key)}")
    print(f"reasoning_split: {args.reasoning_split}")
    print("Sending request...")

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read().decode("utf-8")
            response_json = json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP ERROR: {exc.code}", file=sys.stderr)
        print(body, file=sys.stderr)
        if exc.code == 401:
            print("\nPossible 401 causes:", file=sys.stderr)
            print("- The API key is wrong or expired.", file=sys.stderr)
            print("- The base URL does not match your account region/console.", file=sys.stderr)
            print("- Try global: https://api.minimax.io/v1", file=sys.stderr)
            print("- Try China:  https://api.minimaxi.com/v1", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"NETWORK ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI surface
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        return 1

    content_text, reasoning_text = extract_text(response_json)

    print("Request succeeded.")
    if content_text:
        print("\nAssistant content:")
        print(content_text)
    if reasoning_text:
        print("\nReasoning content:")
        print(reasoning_text)
    if args.show_raw:
        print("\nRaw JSON:")
        print(json.dumps(response_json, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
