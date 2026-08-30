"""Opt-in warm CUDA benchmark for the Transformers rollback adapter.

Run with:
    RUN_LOCAL_GRANITE_TESTS=1 .venv/bin/python scripts/benchmark_granite_query_rewriter.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.model_config import GRANITE_QUERY_REWRITE, QUERY_REWRITER
from backend.providers.granite_query_rewriter import GraniteQueryRewriter
from backend.rag.query_rewrite_contract import ConversationPair, QueryRewritePrompt


RUNS = int(os.getenv("GRANITE_BENCHMARK_RUNS", "20"))
COMPARISON_RUNS = int(os.getenv("GRANITE_PREFILL_COMPARISON_RUNS", "5"))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def prompt() -> QueryRewritePrompt:
    return QueryRewritePrompt(
        "benchmark",
        original_query="But is he more likely to get fleas because of that?",
        conversation_pairs=(
            ConversationPair(
                "rex",
                "I have two pets, a dog named Rex and a cat named Lucy. Rex spends a lot "
                "of time in the backyard and outdoors, and Lucy is always inside.",
                "Rex must love exploring outside, while Lucy enjoys her indoor life.",
            ),
        ),
    )


def raw_generation(adapter: GraniteQueryRewriter, response_prefill: str) -> tuple[int, int, float]:
    messages = adapter._messages(prompt(), len(prompt().conversation_pairs))
    rendered = adapter.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ) + response_prefill
    encoded = adapter.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(
        adapter.config.device
    )
    input_tokens = int(encoded["input_ids"].shape[-1])
    started = time.perf_counter()
    with adapter._torch.inference_mode():
        output = adapter.model.generate(
            **encoded,
            max_new_tokens=adapter.config.max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=adapter.tokenizer.eos_token_id,
            pad_token_id=adapter.tokenizer.pad_token_id,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    generated_tokens = int(output.shape[-1]) - input_tokens
    return input_tokens, generated_tokens, elapsed_ms


def main() -> None:
    if os.getenv("RUN_LOCAL_GRANITE_TESTS") != "1":
        raise SystemExit("Set RUN_LOCAL_GRANITE_TESTS=1 to load the local 6.3 GiB checkpoint")
    started = time.perf_counter()
    adapter = GraniteQueryRewriter()
    startup_seconds = time.perf_counter() - started

    latencies: list[float] = []
    json_successes = 0
    adapter._torch.cuda.reset_peak_memory_stats()
    for _ in range(RUNS):
        run_started = time.perf_counter()
        adapter.complete(prompt(), model=QUERY_REWRITER.model)
        latencies.append((time.perf_counter() - run_started) * 1000)
        json_successes += int(bool(adapter.last_diagnostics and adapter.last_diagnostics.strict_json))

    prefilled_runs = [
        raw_generation(adapter, GRANITE_QUERY_REWRITE.response_prefill)
        for _ in range(COMPARISON_RUNS)
    ]
    unprefilled_runs = [
        raw_generation(adapter, "") for _ in range(COMPARISON_RUNS)
    ]

    def comparison_summary(samples: list[tuple[int, int, float]]) -> dict[str, float]:
        return {
            "input_tokens": samples[-1][0],
            "median_generated_tokens": statistics.median(item[1] for item in samples),
            "median_latency_ms": round(statistics.median(item[2] for item in samples), 3),
        }

    prefilled = comparison_summary(prefilled_runs)
    unprefilled = comparison_summary(unprefilled_runs)
    torch = adapter._torch
    report = {
        "model_path": str(Path(GRANITE_QUERY_REWRITE.model_path)),
        "startup_seconds": round(startup_seconds, 3),
        "warm_latency_ms": {
            "p50": round(statistics.median(latencies), 3),
            "p95": round(percentile(latencies, 0.95), 3),
        },
        "rendered_input_tokens": adapter.last_diagnostics.rendered_input_tokens,
        "generated_tokens": adapter.last_diagnostics.generated_tokens,
        "json_success_rate": json_successes / RUNS,
        "cuda_memory_bytes": {
            "current": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
            "peak": int(torch.cuda.max_memory_allocated()),
        },
        "prefill_comparison": {
            "with_prefill": prefilled,
            "without_prefill": unprefilled,
        },
    }
    print(json.dumps(report, indent=2))
    if report["warm_latency_ms"]["p95"] >= 2000:
        raise SystemExit("Warm p95 rewrite latency violates the two-second TTFT envelope")
    if prefilled["median_generated_tokens"] >= unprefilled["median_generated_tokens"]:
        raise SystemExit("Response prefill did not reduce generated-token count")
    if prefilled["median_latency_ms"] >= unprefilled["median_latency_ms"]:
        raise SystemExit("Response prefill did not reduce median generation latency")


if __name__ == "__main__":
    main()
