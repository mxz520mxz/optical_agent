"""
LLM provider abstraction for the optical lens design agent.

Supports:
  - anthropic : Anthropic Claude API (default)
  - openai    : OpenAI ChatGPT / GPT-4 series
  - local     : Local models via OpenAI-compatible API (Ollama, LM Studio, etc.)

Usage:
  client = make_client("anthropic")
  resp   = call_llm(client, "anthropic", "claude-opus-4-6", system, messages, tools, 8192)
"""

import json
import os
from typing import Any

# ── Default model names per provider ──────────────────────────────────────────

DEFAULT_MODELS = {
    "anthropic": 'claude-3-5-sonnet-20241022', #"claude-opus-4-6"
    "openai":    "gpt-4o",
    "local":     "qwen2.5:72b",   # override via --model
}


# ── Client factory ─────────────────────────────────────────────────────────────

def make_client(provider: str, api_key: str | None = None, base_url: str | None = None):
    """Create a provider-specific API client."""
    if provider == "anthropic":
        import anthropic as _anthropic
        kwargs = {}
        if api_key:
            
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        return _anthropic.Anthropic(**kwargs)

    elif provider in ("openai", "local"):
        from openai import OpenAI
        kwargs: dict[str, Any] = {}
        if api_key:
            print('api_key:',api_key)
            kwargs["api_key"] = api_key
        elif provider == "local":
            # Local models typically don't need a real key
            kwargs["api_key"] = os.environ.get("OPENAI_API_KEY", "local")
        if base_url:
            print('base_url:',base_url)
            kwargs["base_url"] = base_url
        elif provider == "local":
            kwargs["base_url"] = "http://localhost:11434/v1"
       
        return OpenAI(**kwargs)
        # return client

    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose from: anthropic, openai, local")


# ── Tool schema conversion ─────────────────────────────────────────────────────

def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


# ── Message history helpers ────────────────────────────────────────────────────

def append_assistant_message(messages: list[dict], provider: str, raw_response) -> None:
    """Append the assistant turn to the message history (in-place)."""
    if provider == "anthropic":
        messages.append({"role": "assistant", "content": raw_response.content})
    else:
        # OpenAI: append the message object directly
        msg = raw_response.choices[0].message
        
        assistant_msg = {
            "role": "assistant",
            "content": msg.content,
        }
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            
        messages.append(assistant_msg)


def append_tool_results(
    messages: list[dict],
    provider: str,
    tool_results: list[dict],
) -> None:
    """
    Append tool results to message history.

    tool_results format (provider-agnostic):
        [{"id": str, "name": str, "content": str}, ...]
    """
    if provider == "anthropic":
        anthropic_results = [
            {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"]}
            for r in tool_results
        ]
        messages.append({"role": "user", "content": anthropic_results})
    else:
        # OpenAI: one message per tool call with role="tool"
        for r in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": r["id"],
                "content": r["content"],
            })


# ── Unified LLM call ───────────────────────────────────────────────────────────

def call_llm(
    client,
    provider: str,
    model: str,
    system: str,
    messages: list[dict],
    tools: list[dict],
    max_tokens: int = 8192,
) -> dict:
    """
    Call the LLM and return a normalized response dict:

        {
            "stop_reason": "end_turn" | "tool_use",
            "text":        str,           # concatenated text output (may be "")
            "tool_calls":  [              # empty if stop_reason == "end_turn"
                {"id": str, "name": str, "input": dict}
            ],
            "raw": <raw provider response>
        }
    """
    if provider == "anthropic":
        return _call_anthropic(client, model, system, messages, tools, max_tokens)
    else:
        print('we will call openai!!')
        return _call_openai(client, provider, model, system, messages, tools, max_tokens)


def _call_anthropic(client, model, system, messages, tools, max_tokens):
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=tools,
        messages=messages,
    )

    text = ""
    tool_calls = []
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text
        elif block.type == "tool_use":
            tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

    stop_reason = "tool_use" if response.stop_reason == "tool_use" else "end_turn"
    return {"stop_reason": stop_reason, "text": text, "tool_calls": tool_calls, "raw": response}


def _call_openai(client, provider, model, system, messages, tools, max_tokens):
    openai_tools = _anthropic_tools_to_openai(tools)

    # Build message list with system injected at front
    oai_messages = [{"role": "system", "content": system}] + _convert_messages_to_openai(messages)

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        tools=openai_tools,
        tool_choice="auto",
        messages=oai_messages,
        parallel_tool_calls=False,
    )

    choice = response.choices[0]
    msg = choice.message

    text = msg.content or ""
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "input": args})

    finish = choice.finish_reason
    stop_reason = "tool_use" if finish == "tool_calls" else "end_turn"
    return {"stop_reason": stop_reason, "text": text, "tool_calls": tool_calls, "raw": response}

def _convert_messages_to_openai(messages: list[dict]) -> list[dict]:
    """
    Convert Anthropic-format message history to OpenAI format.
    """
    result = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        # 🎯 新增修复：如果是标准的 OpenAI 工具调用或工具结果，原样保留！
        if "tool_calls" in msg or "tool_call_id" in msg:
            result.append(msg)
            continue

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            # Already in OpenAI format
            result.append(msg)
            continue

        # Anthropic content block list (保留你原来的这部分代码)
        text_parts = []
        tool_calls_oai = []
        tool_results_oai = []

        for block in content:
            # Pydantic model (from Anthropic SDK) or dict
            btype = getattr(block, "type", None) or block.get("type", "")

            if btype == "text":
                text_parts.append(getattr(block, "text", block.get("text", "")))

            elif btype == "tool_use":
                bid = getattr(block, "id", block.get("id", ""))
                bname = getattr(block, "name", block.get("name", ""))
                binput = getattr(block, "input", block.get("input", {}))
                tool_calls_oai.append({
                    "id": bid,
                    "type": "function",
                    "function": {"name": bname, "arguments": json.dumps(binput)},
                })

            elif btype == "tool_result":
                tid = getattr(block, "tool_use_id", block.get("tool_use_id", ""))
                tcontent = getattr(block, "content", block.get("content", ""))
                if isinstance(tcontent, list):
                    tcontent = " ".join(
                        getattr(b, "text", b.get("text", "")) for b in tcontent
                    )
                tool_results_oai.append({
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": tcontent,
                })

        if role == "assistant":
            out: dict[str, Any] = {"role": "assistant", "content": " ".join(text_parts)}
            if tool_calls_oai:
                out["tool_calls"] = tool_calls_oai
            result.append(out)
        elif role == "user":
            if text_parts:
                result.append({"role": "user", "content": " ".join(text_parts)})

            # tool_result blocks become role=tool messages
            for tr in tool_results_oai:
                result.append(tr)

    return result