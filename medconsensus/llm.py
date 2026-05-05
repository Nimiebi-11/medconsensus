from __future__ import annotations

import json
import os
from typing import TypeVar
from urllib import request

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def llm_enabled() -> bool:
    return os.getenv("MEDCONSENSUS_USE_LLM", "true").lower() in {"1", "true", "yes", "on"}


def configured_provider() -> str:
    return os.getenv("MEDCONSENSUS_LLM_PROVIDER", "anthropic").lower()


def build_llm_client() -> AnthropicLLMClient | None:
    if not llm_enabled():
        return None
    if configured_provider() != "anthropic":
        return None
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    return AnthropicLLMClient()


class AnthropicLLMClient:
    """Small structured-output client for the LLM-first MedConsensus path."""

    provider = "anthropic"

    def complete_json(self, system_prompt: str, user_payload: dict, output_model: type[T]) -> T:
        payload = {
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            "max_tokens": 1800,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            **user_payload,
                            "output_schema": output_model.model_json_schema(),
                            "format": "Return only valid JSON. No markdown. No prose outside JSON.",
                        },
                        indent=2,
                    ),
                }
            ],
        }
        req = request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
        return output_model.model_validate_json(text)
