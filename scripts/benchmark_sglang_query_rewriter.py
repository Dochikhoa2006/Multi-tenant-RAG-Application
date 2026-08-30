"""Opt-in latency benchmark for the deployed SGLang Granite worker."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from statistics import median
from time import perf_counter

from backend.model_config import QUERY_REWRITER
from backend.providers.sglang_query_rewriter import SGLangGraniteQueryRewriter
from backend.rag.query_rewrite_contract import ConversationPair, QueryRewritePrompt


@dataclass(frozen=True)
class Measurement:
    concurrency: int
    requested_input_tokens: int
    latency_ms: float
    rendered_input_tokens: int
    generated_tokens: int
    cached_prompt_tokens: int | None
    strict_json: bool


@dataclass(frozen=True)
class PrefillMeasurement:
    prefilled: bool
    latency_ms: float
    generated_tokens: int
    strict_json: bool


@dataclass(frozen=True)
class StreamMeasurement:
    ttft_ms: float
    total_latency_ms: float
    strict_json: bool


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _prompt(size: int, sequence: int) -> QueryRewritePrompt:
    # Final sizes are reported from the Granite tokenizer; this text only selects
    # representative short/medium/maximum prompt bands.
    words = max(1, size // 4)
    context = " ".join(f"context{index % 31}" for index in range(words))
    pair = ConversationPair(
        object_id=f"benchmark-{sequence}",
        question=f"What did the prior discussion cover? {context}",
        answer=f"It covered this interview topic. {context}",
    )
    return QueryRewritePrompt(
        "benchmark P1",
        original_query=f"How does that affect my next interview step {sequence}?",
        conversation_pairs=(pair,),
    )


def _one(
    adapter: SGLangGraniteQueryRewriter,
    concurrency: int,
    size: int,
    sequence: int,
) -> Measurement:
    started = perf_counter()
    adapter.complete(_prompt(size, sequence), model=QUERY_REWRITER.model)
    latency_ms = (perf_counter() - started) * 1000.0
    diagnostics = adapter.current_diagnostics
    if diagnostics is None:
        raise RuntimeError("SGLang adapter did not publish diagnostics")
    return Measurement(
        concurrency=concurrency,
        requested_input_tokens=size,
        latency_ms=latency_ms,
        rendered_input_tokens=diagnostics.rendered_input_tokens,
        generated_tokens=diagnostics.generated_tokens,
        cached_prompt_tokens=diagnostics.cached_prompt_tokens,
        strict_json=diagnostics.strict_json,
    )


_FULL_RESPONSE_REGEX = (
    r'\{"rewritten_question":"'
    r'(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9A-Fa-f]{4}))+'
    r'"\}\s*'
)


def _prefill_comparison(
    adapter: SGLangGraniteQueryRewriter,
    *,
    prefilled: bool,
    sequence: int,
) -> PrefillMeasurement:
    prompt = _prompt(512, sequence)
    messages, _, _ = adapter._bounded_messages(prompt)
    body = adapter._request_body(messages)
    if not prefilled:
        body["messages"] = messages[:-1]
        body.pop("continue_final_message", None)
        if adapter.sglang_config.constrained_output:
            body["regex"] = _FULL_RESPONSE_REGEX
    started = perf_counter()
    payload = adapter._send(body)
    latency_ms = (perf_counter() - started) * 1000.0
    try:
        choices = payload["choices"]
        content = choices[0]["message"]["content"]  # type: ignore[index]
        generated_tokens = payload["usage"]["completion_tokens"]  # type: ignore[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise RuntimeError("SGLang benchmark received a malformed response") from exc
    if not isinstance(content, str) or not isinstance(generated_tokens, int):
        raise RuntimeError("SGLang benchmark received malformed output or usage")
    assembled = (
        adapter.granite_config.response_prefill + content if prefilled else content
    )
    strict_json = False
    try:
        parsed = json.loads(assembled)
        strict_json = (
            isinstance(parsed, dict)
            and set(parsed) == {"rewritten_question"}
            and isinstance(parsed["rewritten_question"], str)
            and bool(parsed["rewritten_question"].strip())
        )
    except json.JSONDecodeError:
        pass
    return PrefillMeasurement(
        prefilled=prefilled,
        latency_ms=latency_ms,
        generated_tokens=generated_tokens,
        strict_json=strict_json,
    )


def _stream_probe(
    adapter: SGLangGraniteQueryRewriter,
    *,
    sequence: int,
) -> StreamMeasurement:
    prompt = _prompt(512, sequence)
    messages, _, _ = adapter._bounded_messages(prompt)
    body = adapter._request_body(messages)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    headers = {"Content-Type": "application/json"}
    if adapter.sglang_config.api_key:
        headers["Authorization"] = f"Bearer {adapter.sglang_config.api_key}"
    stream_method = getattr(adapter.client, "stream", None)
    if not callable(stream_method):
        raise RuntimeError("benchmark transport does not support HTTP streaming")

    started = perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    with stream_method(
        "POST",
        f"{adapter.sglang_config.base_url}/chat/completions",
        json=body,
        headers=headers,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
                choices = event.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content")
                    if content:
                        if not isinstance(content, str):
                            raise TypeError
                        if first_token_at is None:
                            first_token_at = perf_counter()
                        pieces.append(content)
            except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("SGLang streaming benchmark response is malformed") from exc
    finished = perf_counter()
    if first_token_at is None:
        raise RuntimeError("SGLang streaming benchmark returned no content")
    assembled = adapter.granite_config.response_prefill + "".join(pieces)
    try:
        parsed = json.loads(assembled)
        strict_json = (
            isinstance(parsed, dict)
            and set(parsed) == {"rewritten_question"}
            and isinstance(parsed["rewritten_question"], str)
            and bool(parsed["rewritten_question"].strip())
        )
    except json.JSONDecodeError:
        strict_json = False
    return StreamMeasurement(
        ttft_ms=(first_token_at - started) * 1000.0,
        total_latency_ms=(finished - started) * 1000.0,
        strict_json=strict_json,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-per-case", type=int, default=40)
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--input-tokens", default="128,512,1024,2048")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--prefill-comparison-samples", type=int, default=20)
    parser.add_argument("--stream-ttft-samples", type=int, default=20)
    parser.add_argument("--gpu", default=os.getenv("MODAL_SGLANG_GPU", "L40S"))
    parser.add_argument("--gpu-price-per-second", type=float, default=0.000542)
    parser.add_argument("--baseline-p95-ms", type=float, default=None)
    arguments = parser.parse_args()
    if min(
        arguments.requests_per_case,
        arguments.warmup,
        arguments.prefill_comparison_samples,
        arguments.stream_ttft_samples,
    ) <= 0:
        raise ValueError("benchmark request counts must be positive")
    concurrency_values = [int(value) for value in arguments.concurrency.split(",")]
    input_sizes = [int(value) for value in arguments.input_tokens.split(",")]
    if min(*concurrency_values, *input_sizes) <= 0:
        raise ValueError("concurrency and input token sizes must be positive")

    adapter = SGLangGraniteQueryRewriter()
    for index in range(arguments.warmup):
        adapter.complete(_prompt(128, -index), model=QUERY_REWRITER.model)

    measurements: list[Measurement] = []
    elapsed_by_case: dict[tuple[int, int], float] = {}
    for concurrency in concurrency_values:
        for size in input_sizes:
            case_started = perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                measurements.extend(
                    executor.map(
                        lambda sequence: _one(
                            adapter,
                            concurrency,
                            size,
                            sequence,
                        ),
                        range(arguments.requests_per_case),
                    )
                )
            elapsed_by_case[(concurrency, size)] = perf_counter() - case_started

    comparisons = [
        _prefill_comparison(
            adapter,
            prefilled=prefilled,
            sequence=sequence,
        )
        for sequence in range(arguments.prefill_comparison_samples)
        for prefilled in (True, False)
    ]
    stream_measurements = [
        _stream_probe(adapter, sequence=sequence)
        for sequence in range(arguments.stream_ttft_samples)
    ]

    cases: list[dict[str, object]] = []
    for concurrency in concurrency_values:
        for size in input_sizes:
            selected = [
                item
                for item in measurements
                if item.concurrency == concurrency
                and item.requested_input_tokens == size
            ]
            latencies = [item.latency_ms for item in selected]
            wall_seconds = elapsed_by_case[(concurrency, size)]
            cached_tokens = sum(item.cached_prompt_tokens or 0 for item in selected)
            rendered_tokens = sum(item.rendered_input_tokens for item in selected)
            cases.append(
                {
                    "concurrency": concurrency,
                    "requested_input_tokens": size,
                    "median_rendered_input_tokens": median(
                        item.rendered_input_tokens for item in selected
                    ),
                    "p50_latency_ms": round(median(latencies), 3),
                    "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
                    "mean_generated_tokens": round(
                        sum(item.generated_tokens for item in selected) / len(selected),
                        3,
                    ),
                    "json_success_rate": round(
                        sum(item.strict_json for item in selected) / len(selected),
                        4,
                    ),
                    "cached_prompt_token_ratio": round(
                        cached_tokens / rendered_tokens,
                        4,
                    ),
                    "throughput_rewrites_per_second": round(
                        len(selected) / wall_seconds,
                        3,
                    ),
                    "estimated_gpu_cost_per_1000_rewrites_usd": round(
                        wall_seconds
                        * arguments.gpu_price_per_second
                        * 1000.0
                        / len(selected),
                        6,
                    ),
                }
            )
    overall_p95 = _percentile(
        [measurement.latency_ms for measurement in measurements],
        0.95,
    )
    report: dict[str, object] = {
        "gpu": arguments.gpu,
        "cases": cases,
        "overall_p95_ms": round(overall_p95, 3),
        "rewrite_p95_gate_ms": 500.0,
        "passes_rewrite_gate": overall_p95 <= 500.0,
    }
    prefilled = [item for item in comparisons if item.prefilled]
    unprefilled = [item for item in comparisons if not item.prefilled]
    prefilled_p95 = _percentile([item.latency_ms for item in prefilled], 0.95)
    unprefilled_p95 = _percentile([item.latency_ms for item in unprefilled], 0.95)
    prefilled_tokens = sum(item.generated_tokens for item in prefilled) / len(prefilled)
    unprefilled_tokens = sum(item.generated_tokens for item in unprefilled) / len(
        unprefilled
    )
    report["response_prefill_comparison"] = {
        "samples_per_mode": arguments.prefill_comparison_samples,
        "prefilled_p95_ms": round(prefilled_p95, 3),
        "unprefilled_p95_ms": round(unprefilled_p95, 3),
        "prefilled_mean_generated_tokens": round(prefilled_tokens, 3),
        "unprefilled_mean_generated_tokens": round(unprefilled_tokens, 3),
        "prefilled_json_success_rate": round(
            sum(item.strict_json for item in prefilled) / len(prefilled),
            4,
        ),
        "unprefilled_json_success_rate": round(
            sum(item.strict_json for item in unprefilled) / len(unprefilled),
            4,
        ),
        "prefill_generates_fewer_tokens": prefilled_tokens < unprefilled_tokens,
        "prefill_is_no_slower_at_p95": prefilled_p95 <= unprefilled_p95,
    }
    stream_ttfts = [item.ttft_ms for item in stream_measurements]
    stream_totals = [item.total_latency_ms for item in stream_measurements]
    report["streaming_server_probe"] = {
        "samples": arguments.stream_ttft_samples,
        "application_visible_server_ttft_p50_ms": round(median(stream_ttfts), 3),
        "application_visible_server_ttft_p95_ms": round(
            _percentile(stream_ttfts, 0.95),
            3,
        ),
        "total_latency_p50_ms": round(median(stream_totals), 3),
        "total_latency_p95_ms": round(_percentile(stream_totals, 0.95), 3),
        "json_success_rate": round(
            sum(item.strict_json for item in stream_measurements)
            / len(stream_measurements),
            4,
        ),
    }
    if arguments.baseline_p95_ms is not None:
        report["baseline_p95_ms"] = arguments.baseline_p95_ms
        report["p95_reduction_fraction"] = round(
            1.0 - overall_p95 / arguments.baseline_p95_ms,
            4,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    adapter.close()


if __name__ == "__main__":
    main()
