from __future__ import annotations

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import asyncio
import json
from pathlib import Path
import threading
import time

import pytest

from backend.model_config import (
    GRANITE_QUERY_REWRITE,
    QUERY_REWRITER,
    SESSION_TITLE_GENERATOR,
)
from backend.providers.granite_query_rewriter import (
    GraniteCheckpointError,
    GraniteInferenceError,
    GraniteQueryRewriter,
    RoleRoutingLLMClient,
    validate_granite_checkpoint,
)
from backend.rag.query_rewrite_contract import ConversationPair, QueryRewritePrompt


class FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.shape = (1, len(values))
        self.devices: list[str] = []

    def to(self, device: str) -> FakeTensor:
        self.devices.append(device)
        return self


class FakeEncoding(dict[str, FakeTensor]):
    def to(self, device: str) -> FakeEncoding:
        for value in self.values():
            value.to(device)
        return self


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, continuation: str, *, token_mode: str = "words") -> None:
        self.continuation = continuation
        self.token_mode = token_mode
        self.template_calls: list[list[dict[str, str]]] = []
        self.encoded_texts: list[str] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.template_calls.append(messages)
        turns = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        return turns + "<assistant>"

    def __call__(self, text: str, **kwargs: object) -> FakeEncoding:
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
        self.encoded_texts.append(text)
        count = len(text) if self.token_mode == "characters" else len(text.split())
        return FakeEncoding(input_ids=FakeTensor(list(range(count))))

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return self.continuation


class FakeModel:
    def __init__(self, generated_tokens: int = 4, *, delay: float = 0.0) -> None:
        self.generated_tokens = generated_tokens
        self.delay = delay
        self.calls: list[dict[str, object]] = []
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.calls.append(kwargs)
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, FakeTensor)
            input_count = input_ids.shape[-1]
            return [list(range(input_count + self.generated_tokens))]
        finally:
            with self._guard:
                self.active -= 1


def _config(**changes: object):
    return replace(GRANITE_QUERY_REWRITE, warmup=False, **changes)


def _prompt(*pairs: ConversationPair, query: str = "What about it?") -> QueryRewritePrompt:
    return QueryRewritePrompt(
        "legacy P1 prompt",
        original_query=query,
        conversation_pairs=pairs,
    )


def _rewriter(continuation: str, **config_changes: object):
    tokenizer = FakeTokenizer(continuation)
    model = FakeModel()
    adapter = GraniteQueryRewriter(
        config=_config(**config_changes),
        tokenizer=tokenizer,
        model=model,
    )
    return adapter, tokenizer, model


def test_structured_messages_preserve_mmr_order_and_end_with_latest_query() -> None:
    first = ConversationPair("one", "Who is Rex?", "Rex is my dog.")
    second = ConversationPair("two", "What has fleas?", "Rex has fleas.")
    adapter, tokenizer, model = _rewriter('What causes fleas on Rex?"}')

    result = adapter.complete(_prompt(first, second), model=QUERY_REWRITER.model)

    assert result == "What causes fleas on Rex?"
    assert tokenizer.template_calls == [[
        {"role": "user", "content": "Who is Rex?"},
        {"role": "assistant", "content": "Rex is my dog."},
        {"role": "user", "content": "What has fleas?"},
        {"role": "assistant", "content": "Rex has fleas."},
        {"role": "user", "content": "What about it?"},
    ]]
    assert tokenizer.encoded_texts[-1].endswith(
        "<assistant>" + GRANITE_QUERY_REWRITE.response_prefill
    )
    call = model.calls[0]
    assert call["max_new_tokens"] == 128
    assert call["do_sample"] is False
    assert call["num_beams"] == 1
    assert call["use_cache"] is True


def test_whole_lowest_ranked_pairs_are_removed_until_rendered_input_fits() -> None:
    pairs = (
        ConversationPair("one", "first context words", "first answer words"),
        ConversationPair("two", "second context words", "second answer words"),
    )
    adapter, tokenizer, _ = _rewriter(
        'Standalone question"}',
        max_input_tokens=70,
    )
    tokenizer.token_mode = "characters"

    adapter.complete(_prompt(*pairs, query="latest"), model=QUERY_REWRITER.model)

    assert len(tokenizer.template_calls) == 3
    assert [message["content"] for message in tokenizer.template_calls[-1]] == [
        "latest"
    ]
    assert adapter.last_diagnostics is not None
    assert adapter.last_diagnostics.dropped_conversation_pairs == 2
    assert adapter.last_diagnostics.rendered_input_tokens <= 70


def test_latest_query_that_cannot_fit_is_rejected_without_truncation() -> None:
    adapter, _, model = _rewriter(
        'unused"}',
        max_input_tokens=1,
    )

    with pytest.raises(GraniteInferenceError, match="Latest query"):
        adapter.complete(_prompt(query="too large"), model=QUERY_REWRITER.model)
    assert model.calls == []


@pytest.mark.parametrize(
    ("continuation", "expected"),
    [
        ('Clear standalone query"}', "Clear standalone query"),
        ('Explain \\"Rex\\" and fleas"}', 'Explain "Rex" and fleas'),
    ],
)
def test_strict_json_returns_only_rewritten_question(
    continuation: str,
    expected: str,
) -> None:
    adapter, _, _ = _rewriter(continuation)
    assert adapter.complete(_prompt(), model=QUERY_REWRITER.model) == expected
    assert adapter.last_diagnostics is not None
    assert adapter.last_diagnostics.strict_json is True


@pytest.mark.parametrize(
    "continuation",
    [
        'query","extra":"field"}',
        'query"} trailing',
        "unfinished query",
        'query",}',
        'query","wrong_key":"value"}',
    ],
)
def test_contract_failures_return_complete_prefilled_output(
    continuation: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter, _, _ = _rewriter(continuation)
    result = adapter.complete(_prompt(), model=QUERY_REWRITER.model)
    assert result == GRANITE_QUERY_REWRITE.response_prefill + continuation
    assert continuation not in caplog.text
    assert "strict JSON contract" in caplog.text


@pytest.mark.parametrize("continuation", ["", "   ", '"}', "}\n"])
def test_empty_or_scaffold_only_continuation_fails(continuation: str) -> None:
    adapter, _, _ = _rewriter(continuation)
    with pytest.raises(GraniteInferenceError, match="meaningful continuation"):
        adapter.complete(_prompt(), model=QUERY_REWRITER.model)


def test_inference_is_serialized_by_process_local_lock() -> None:
    tokenizer = FakeTokenizer('query"}')
    model = FakeModel(delay=0.03)
    adapter = GraniteQueryRewriter(
        config=_config(),
        tokenizer=tokenizer,
        model=model,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(adapter.complete, _prompt(), model=QUERY_REWRITER.model)
            for _ in range(2)
        ]
        assert [future.result() for future in futures] == ["query", "query"]
    assert model.max_active == 1


def test_configured_warmup_runs_one_greedy_cached_token() -> None:
    tokenizer = FakeTokenizer('query"}')
    model = FakeModel()
    GraniteQueryRewriter(
        config=replace(GRANITE_QUERY_REWRITE, warmup=True),
        tokenizer=tokenizer,
        model=model,
    )
    assert len(model.calls) == 1
    assert model.calls[0]["max_new_tokens"] == 1
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["use_cache"] is True


class Delegate:
    def __init__(self) -> None:
        self.complete_calls: list[tuple[str, dict[str, object]]] = []
        self.stream_calls: list[tuple[str, dict[str, object]]] = []

    def complete(self, prompt: str, **kwargs: object) -> str:
        self.complete_calls.append((prompt, kwargs))
        return "delegated"

    def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        self.stream_calls.append((prompt, kwargs))

        async def tokens() -> AsyncIterator[str]:
            yield "answer"

        return tokens()


class RecordingGranite:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def complete(self, prompt: str, **kwargs: object) -> str:
        self.calls.append((prompt, kwargs))
        if not isinstance(prompt, QueryRewritePrompt):
            raise TypeError("structured prompt required")
        return "granite"


def test_role_router_changes_only_query_rewriting() -> None:
    delegate = Delegate()
    granite = RecordingGranite()
    router = RoleRoutingLLMClient(delegate, granite)  # type: ignore[arg-type]
    prompt = _prompt()

    assert router.complete(prompt, model=QUERY_REWRITER.model) == "granite"
    assert (
        router.complete("title", model=SESSION_TITLE_GENERATOR.model)
        == "delegated"
    )
    async def consume() -> list[str]:
        stream = router.stream("answer", model="GPT-5.1")
        return [token async for token in stream]  # type: ignore[union-attr]

    assert asyncio.run(consume()) == ["answer"]
    assert len(granite.calls) == 1
    assert [call[0] for call in delegate.complete_calls] == ["title"]
    assert [call[0] for call in delegate.stream_calls] == ["answer"]


def test_router_rejects_unstructured_granite_calls() -> None:
    router = RoleRoutingLLMClient(Delegate(), RecordingGranite())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="structured prompt"):
        router.complete("plain P1", model=QUERY_REWRITER.model)


def test_checkpoint_validation_requires_expected_architecture_and_two_shards(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GraniteForCausalLM"],
                "model_type": "granite",
                "dtype": "float16",
            }
        ),
        encoding="utf-8",
    )
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for name in shard_names:
        (tmp_path / name).write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shard_names[0], "b": shard_names[1]}}),
        encoding="utf-8",
    )

    assert validate_granite_checkpoint(tmp_path) == tuple(shard_names)
    (tmp_path / shard_names[1]).unlink()
    with pytest.raises(GraniteCheckpointError, match="missing or invalid"):
        validate_granite_checkpoint(tmp_path)
