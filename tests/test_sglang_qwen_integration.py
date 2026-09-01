"""Opt-in contract smoke test for a provisioned Qwen SGLang worker."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path

import httpx
import pytest

from backend.model_config import QWEN_SGLANG
from backend.providers.sglang_qwen_llm import SGLangQwenLLMClient
from backend.rag.session_title import validate_session_title


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_QWEN_SGLANG_TESTS") != "1",
    reason="requires the opt-in Qwen SGLang worker",
)


def test_qwen_sglang_offline_non_thinking_contract() -> None:
    config = replace(
        QWEN_SGLANG,
        api_key=os.environ["QWEN_SGLANG_API_KEY"],
    )
    model_config = json.loads(
        (Path(config.model_path) / "config.json").read_text(encoding="utf-8")
    )
    assert model_config["max_position_embeddings"] >= 32768

    headers = {"Authorization": f"Bearer {config.api_key}"}
    root = config.base_url.removesuffix("/v1")
    with httpx.Client(
        headers=headers,
        timeout=config.read_timeout_seconds,
    ) as probe:
        assert probe.get(f"{root}/health").is_success
        models = probe.get(f"{config.base_url}/models")
        models.raise_for_status()
        assert config.served_model in {
            item["id"] for item in models.json().get("data", [])
        }

    client = SGLangQwenLLMClient(config)
    try:
        title = client.complete(
            "Return only a three word title for a discussion about RAG retrieval.",
            model=config.served_model,
        )
        validate_session_title(title)
        assert "<think>" not in title

        async def answer(prompt: str) -> str:
            chunks = [
                chunk
                async for chunk in client.stream(
                    prompt,
                    model=config.served_model,
                    reasoning="low",
                    max_output_tokens=64,
                )
            ]
            return "".join(chunks)

        async def run_concurrently() -> tuple[str, str]:
            try:
                first, second = await asyncio.gather(
                    answer(
                        "Answer in one sentence: what is retrieval augmented generation?"
                    ),
                    answer("Answer in one sentence: what is semantic search?"),
                )
                return first, second
            finally:
                await client.aclose()

        first, second = asyncio.run(run_concurrently())
        assert first.strip() and second.strip()
        assert "<think>" not in first and "<think>" not in second
    finally:
        client.close()
