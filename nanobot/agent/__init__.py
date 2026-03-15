"""LLM provider for local OpenAI-compatible APIs (Ollama, vLLM, LM Studio, etc.)."""

import json
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ToolCall:
    """A tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LocalLLMProvider:
    """
    LLM provider using any OpenAI-compatible API.
    
    Works with: Ollama, vLLM, llama.cpp, LM Studio, LocalAI, etc.
    """

    def __init__(self, api_base: str, api_key: str = "no-key", model: str = "default"):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(timeout=300.0)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send a chat completion request."""
        url = f"{self.api_base}/chat/completions"

        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            resp = await self._client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return self._parse(data)
        except httpx.HTTPStatusError as e:
            error_body = e.response.text[:500] if e.response else str(e)
            return LLMResponse(content=f"LLM API error ({e.response.status_code}): {error_body}")
        except Exception as e:
            return LLMResponse(content=f"LLM request failed: {e}")

    def _parse(self, data: dict) -> LLMResponse:
        """Parse OpenAI-format response."""
        choice = data["choices"][0]
        message = choice["message"]

        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(ToolCall(
                    id=tc.get("id", f"call_{len(tool_calls)}"),
                    name=tc["function"]["name"],
                    arguments=args,
                ))

        usage = {}
        if data.get("usage"):
            usage = {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
            }

        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            usage=usage,
        )

    async def close(self):
        await self._client.aclose()
