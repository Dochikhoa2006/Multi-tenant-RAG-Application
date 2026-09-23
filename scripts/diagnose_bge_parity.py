"""Read-only BGE parity on captured pairs; never invokes the RAG pipeline.

Prepare with the application Python (tokenizer + vector-free Weaviate reads).
Run with the isolated export-reference Python (torch 2.8.0/transformers 4.56.2).
Only new diagnostic output files are written. Model loading is offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL = "BAAI/bge-reranker-v2-m3"
REFERENCE_WEIGHTS = "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286"
STATE_HASH = "039fbb21cd61efc027f7bbf5a38a622d2d016d4843ce588b5de202bfac99a55e"
USER = "rag_real_final_v2"
QUERY_IDS = (
    "legacy-000004-379fbbb59f0a0f95",
    "legacy-000001-796d76f0652a3623",
    "legacy-000006-d38735b261f3c0b7",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def content_digest(text: str) -> str:
    raw = text.encode("utf-8")
    return hashlib.sha256(
        b"chat-kp-candidate-text-v1" + len(raw).to_bytes(8, "big") + raw
    ).hexdigest()


def write_new(path: Path, value: object) -> None:
    # Refuse accidental replacement of earlier evidence.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_manifest(root: Path) -> dict:
    manifest = json.loads((root / "onnx-manifest.json").read_text())
    if manifest["model_id"] != MODEL or manifest["revision"] != REVISION:
        raise ValueError("Wrong production model identity")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError("Production artifact hash mismatch")
    return manifest


def decode_candidates(operation: dict, collection: str) -> list[dict]:
    fields = operation["texts"][f"{collection}_candidate_decision_fields"].split("|")
    rows, fingerprints = [], {}
    for page in (1, 2):
        for kind in ("decisions", "text_digests"):
            sample = operation["samples"][f"{collection}_candidate_{kind}_page_{page}"]
            if sample["truncated"] or sample["exact_count"] != len(sample["items"]) or len(sample["items"]) > 32:
                raise ValueError("Incomplete candidate page")
            if kind == "decisions":
                rows.extend(dict(zip(fields, item.split("|"), strict=True)) for item in sample["items"])
            else:
                for item in sample["items"]:
                    identifier, fingerprint = item.split(":", 1)
                    if identifier in fingerprints:
                        raise ValueError("Duplicate fingerprint identity")
                    fingerprints[identifier] = fingerprint
    ceiling = 50 if collection == "knowledge" else 40
    if len(rows) != ceiling or len({r["candidate_id"] for r in rows}) != ceiling:
        raise ValueError("Incomplete or duplicate candidate identities")
    for row in rows:
        identifier = row["candidate_id"]
        if str(UUID(identifier)) != identifier:
            raise ValueError("Malformed candidate UUID")
        row["text_digest"] = fingerprints[identifier]
        row["collection"] = collection
        row["hybrid_rank"] = int(row["hybrid_rank"])
        row["bge_rank"] = int(row["bge_rank"])
        row["captured_logit"] = float.fromhex(row["raw_score_hex"])
        row["floor"] = float.fromhex(row["floor_hex"])
        if not all(math.isfinite(row[k]) for k in ("captured_logit", "floor")):
            raise ValueError("Nonfinite captured score")
        if (row["captured_logit"] >= row["floor"]) != (row["floor_pass"] == "1"):
            raise ValueError("Captured floor decision inconsistent")
    if sorted(r["hybrid_rank"] for r in rows) != list(range(1, ceiling + 1)):
        raise ValueError("Hybrid ranks incomplete")
    if sorted(r["bge_rank"] for r in rows) != list(range(1, ceiling + 1)):
        raise ValueError("BGE ranks incomplete")
    return sorted(rows, key=lambda row: row["hybrid_rank"])


def prepare(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(ROOT))
    from deployment.ragctl import load_dotenv, ENV_PATH
    from deployment.wizard_diagnostic import load_corpus_state
    from deployment.wizard_diagnostic_api import CorpusStorage
    import transformers

    state_path = ROOT / ".local/diagnostics/wizard/corpus-state.json"
    if digest(state_path) != STATE_HASH:
        raise ValueError("Canonical state changed")
    state = load_corpus_state(state_path, USER)
    if state is None or state.active is None or state.pending_replacement or state.pending_cleanup:
        raise ValueError("Corpus is not reusable")
    manifest = validate_manifest(args.model_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    source = [json.loads(line) for line in args.requests.read_text().splitlines()]
    if len(source) != 3 or {r["query_id"] for r in source} != set(QUERY_IDS):
        raise ValueError("Expected exact three-query observation run")
    documents = {d.collection: d for d in state.active.documents}
    config = dict(load_dotenv(ENV_PATH))
    config.update(WEAVIATE_URL="http://127.0.0.1:8080", WEAVIATE_GRPC_PORT="50051", WEAVIATE_GRPC_SECURE="false")
    groups = []
    with CorpusStorage(config, USER) as storage:
        storage.validate_membership(state)
        for qid in QUERY_IDS:  # Q4 first, Q1 second, Q6 last.
            request = next(r for r in source if r["query_id"] == qid)
            op = request["deep_trace"]["operation"]
            if request["status"] != "succeeded" or op["user_id"] != USER:
                raise ValueError("Invalid captured operation")
            for collection in (("knowledge", "policy") if qid == QUERY_IDS[0] else (("knowledge",) if qid == QUERY_IDS[1] else ("policy",))):
                identity = dict(part.split("=", 1) for part in op["texts"][f"{collection}_reranker_identity"].split(";"))
                if identity != dict(model=MODEL, revision=REVISION, onnx="model_fp16.onnx", manifest="onnx-manifest.json", output="logits", max_tokens="512"):
                    raise ValueError("Captured reranker identity mismatch")
                rows = decode_candidates(op, collection)
                doc = documents[collection]
                expected_ids = set(doc.chunk_ids)
                accessor = storage._collection(USER, collection)
                response = accessor._collection.query.fetch_objects_by_ids(
                    [r["candidate_id"] for r in rows], limit=len(rows), include_vector=False,
                    return_properties=["user_id", "document_id", "chunk_id", "raw_text"],
                )
                objects = {}
                for obj in response.objects:
                    p = obj.properties; oid = str(obj.uuid)
                    if oid in objects or oid not in expected_ids or str(p["chunk_id"]) != oid or p["user_id"] != USER or str(p["document_id"]) != doc.wizard_id:
                        raise ValueError("Physical ownership mismatch")
                    objects[oid] = p["raw_text"]
                if set(objects) != {r["candidate_id"] for r in rows}:
                    raise ValueError("Incomplete physical hydration")
                for row in rows:
                    row["text"] = objects[row["candidate_id"]]
                    if content_digest(row["text"]) != row["text_digest"]:
                        raise ValueError("Physical text does not match captured pair")
                    row["document_id"] = doc.wizard_id
                    row["evaluate"] = qid == QUERY_IDS[0] or (
                        qid == QUERY_IDS[1] and "Separate API keys do not multiply" in row["text"]
                    ) or (qid == QUERY_IDS[2] and "do not convert a generated answer into a contractual amendment" in row["text"])
                query = op["texts"]["granite_rewritten_query"]
                batches = []
                for start in range(0, len(rows), 16):
                    batch = rows[start:start + 16]
                    pairs = [(query, r["text"]) for r in batch]
                    encoded = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="np")
                    batches.append({"start": start, "inputs": {k: v.tolist() for k, v in encoded.items()}})
                groups.append({"query_id": qid, "collection": collection, "query": query, "candidates": rows, "batches": batches})
    if [sum(r["evaluate"] for r in g["candidates"]) for g in groups] != [50, 40, 16, 3]:
        raise ValueError("Unexpected supporting-pair count")
    if digest(state_path) != STATE_HASH:
        raise ValueError("Canonical state changed during read")
    write_new(args.output, {"schema_version": "1", "requests_sha256": digest(args.requests), "state_sha256": STATE_HASH,
        "generation_id": state.active.generation_id, "production_manifest": manifest,
        "production_tokenizer_version": transformers.__version__, "groups": groups})
    print("Prepared 90 Q4 pairs, 16 relevant Q1 pairs, 3 relevant Q6 pairs; ownership and digests verified.", flush=True)


def comparison(rows: list[dict], key: str) -> dict:
    if not rows or any(not math.isfinite(float(r[key])) for r in rows):
        raise ValueError("Incomplete/nonfinite comparison")
    ranked = sorted(rows, key=lambda r: (-r[key], r["candidate_id"]))
    captured = sorted(rows, key=lambda r: (-r["captured_logit"], r["candidate_id"]))
    ranks = {r["candidate_id"]: i for i, r in enumerate(ranked, 1)}
    flips = [r["candidate_id"] for r in rows if (r[key] >= r["floor"]) != (r["captured_logit"] >= r["floor"])]
    inversions, tie_changes = 0, 0
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            d1, d2 = a["captured_logit"]-b["captured_logit"], a[key]-b[key]
            inversions += d1*d2 < 0
            tie_changes += (d1 == 0) != (d2 == 0)
    for row in rows:
        row[key + "_rank_in_compared_pool"] = ranks[row["candidate_id"]]
        row[key + "_floor_pass"] = row[key] >= row["floor"]
        row[key + "_minus_captured"] = row[key]-row["captured_logit"]
    return {"count": len(rows), "max_abs_logit_error": max(abs(r[key]-r["captured_logit"]) for r in rows),
        "mean_abs_logit_error": sum(abs(r[key]-r["captured_logit"]) for r in rows)/len(rows),
        "floor_flip_ids": flips, "strict_pairwise_inversions": inversions, "tie_status_changes": tie_changes,
        "max_rank_displacement": max(abs(i-ranks[r["candidate_id"]]) for i,r in enumerate(captured,1))}


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import transformers
    import onnxruntime as ort

    if torch.__version__.split("+")[0] != "2.8.0" or transformers.__version__ != "4.56.2" or ort.__version__ != "1.29.0":
        raise ValueError("Use isolated torch 2.8.0, transformers 4.56.2, ORT 1.29.0")
    if args.reference_dir.name != REVISION or digest(args.reference_dir / "model.safetensors") != REFERENCE_WEIGHTS:
        raise ValueError("Pinned reference checkpoint hash mismatch")
    data = json.loads(args.pairs.read_text())
    if validate_manifest(args.model_dir) != data["production_manifest"]:
        raise ValueError("Production artifact changed since capture")
    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.reference_dir), local_files_only=True, trust_remote_code=False)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        str(args.reference_dir), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).eval().cpu()
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(args.model_dir / "model_fp16.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    report = {"schema_version": "1", "pairs_sha256": digest(args.pairs), "reference_weights_sha256": REFERENCE_WEIGHTS,
        "reference": "BAAI published Transformers sequence-classification raw-logit implementation; pinned export dependency versions; FP32 CPU eager",
        "production": "captured ONNX FP16 CUDA logits; no production execution performed",
        "supplemental": "unchanged FP16 ONNX graph replay on CPU, not CUDA parity",
        "torch": torch.__version__, "transformers": transformers.__version__, "onnxruntime": ort.__version__,
        "production_tokenizer_version": data["production_tokenizer_version"], "groups": []}
    for group in data["groups"]:
        rows = group["candidates"]
        print("Scoring", group["query_id"], group["collection"], flush=True)
        for batch in group["batches"]:
            part = rows[batch["start"]:batch["start"]+16]
            inputs = tokenizer([(group["query"], r["text"]) for r in part], padding=True, truncation=True, max_length=512, return_tensors="np")
            if set(inputs) != set(batch["inputs"]) or any(not np.array_equal(v, batch["inputs"][k]) for k,v in inputs.items()):
                raise ValueError("Reference/production tokenizer inputs differ; scoring aborted")
            indices = [i for i,r in enumerate(part) if r["evaluate"]]
            if not indices:
                continue
            # Preserve each original batch's padding length for supplemental subsets.
            selected = {k: v[indices] for k,v in inputs.items()}
            feed = {m.name: np.asarray(selected[m.name], dtype=np.int64) for m in session.get_inputs()}
            pt = {k: torch.from_numpy(v.copy()) for k,v in selected.items()}
            with torch.inference_mode():
                first = model(**pt, return_dict=True).logits.view(-1).float().numpy()
                repeat = model(**pt, return_dict=True).logits.view(-1).float().numpy()
            onnx = np.asarray(session.run(["logits"], feed)[0], dtype=np.float32).reshape(-1)
            onnx_repeat = np.asarray(session.run(["logits"], feed)[0], dtype=np.float32).reshape(-1)
            if not np.array_equal(first,repeat) or not np.array_equal(onnx,onnx_repeat):
                raise ValueError("Diagnostic repeat was not bitwise deterministic")
            for i, index in enumerate(indices):
                part[index].update(reference_fp32=float(first[i]), onnx_fp16_cpu=float(onnx[i]))
        observed = [r for r in rows if r["evaluate"]]
        report["groups"].append({"query_id": group["query_id"], "collection": group["collection"], "query": group["query"],
            "rank_scope": "complete captured pool" if group["query_id"] == QUERY_IDS[0] else "relevant subset only; original BGE ranks retained separately",
            "tokenizer_inputs_identical": True, "repeat_bitwise_identical": True,
            "reference_difference": comparison(observed,"reference_fp32"), "cpu_onnx_difference": comparison(observed,"onnx_fp16_cpu"), "candidates": observed})
    if validate_manifest(args.model_dir) != data["production_manifest"] or digest(args.reference_dir / "model.safetensors") != REFERENCE_WEIGHTS:
        raise ValueError("Model artifact changed during diagnostic")
    write_new(args.output, report)
    for group in report["groups"]:
        print(group["query_id"], group["collection"], json.dumps(group["reference_difference"]), flush=True)


def main() -> None:
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    prepare_cli = sub.add_parser("prepare")
    prepare_cli.add_argument("--requests", type=Path, required=True)
    run_cli = sub.add_parser("run")
    run_cli.add_argument("--pairs", type=Path, required=True)
    run_cli.add_argument("--reference-dir", type=Path, required=True)
    for parser in (prepare_cli, run_cli):
        parser.add_argument("--model-dir", type=Path, default=ROOT / "models/bge-reranker-v2-m3-onnx")
        parser.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Output already exists")
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
