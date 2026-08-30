"""Exact eight-step wizard save and re-embedding pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
import math
from uuid import UUID

from backend.mappings._common import positive_paragraph_id
from backend.weaviate_client.models import ChunkRecord, PartialParagraphUpdateError
from backend.wizard.errors import WizardSaveError, WizardSaveRecoveryError
from backend.wizard.runtime import WizardRuntime, resolve_runtime


@dataclass(frozen=True)
class _SavedParagraph:
    paragraph_id: int
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _ParagraphUnion:
    paragraph_ids: tuple[int, ...]
    current_text: str


@dataclass(frozen=True)
class _StagedChunk:
    chunk_id: str
    raw_text: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class _ParagraphNode:
    raw_text: str
    chunks: tuple[_StagedChunk, ...]
    old_paragraph_id: int | None


Opcode = tuple[str, int, int, int, int]


def _saved_paragraphs(paragraph_data: dict[int, str]) -> list[_SavedParagraph]:
    paragraphs: list[_SavedParagraph] = []
    offset = 0
    for paragraph_id, text in paragraph_data.items():
        end = offset + len(text)
        paragraphs.append(
            _SavedParagraph(
                paragraph_id=paragraph_id,
                text=text,
                start=offset,
                end=end,
            )
        )
        offset = end
    return paragraphs


def _validated_modified_ids(
    modified_paragraph_ids: object,
    existing_ids: set[int],
) -> set[int]:
    if modified_paragraph_ids is None:
        return set()
    if not isinstance(modified_paragraph_ids, list):
        raise TypeError("modified_paragraph_ids must be a list or None")
    validated: set[int] = set()
    for raw_paragraph_id in modified_paragraph_ids:
        paragraph_id = positive_paragraph_id(raw_paragraph_id)
        if paragraph_id not in existing_ids:
            raise ValueError(
                f"modified paragraph_id {paragraph_id} does not exist in the document"
            )
        validated.add(paragraph_id)
    return validated


def _paragraphs_overlapping(
    paragraphs: list[_SavedParagraph],
    start: int,
    end: int,
) -> set[int]:
    return {
        paragraph.paragraph_id
        for paragraph in paragraphs
        if paragraph.start < end and paragraph.end > start
    }


def _paragraphs_at_insertion(
    paragraphs: list[_SavedParagraph],
    position: int,
) -> set[int]:
    if position <= 0:
        return {paragraphs[0].paragraph_id}
    if position >= paragraphs[-1].end:
        return {paragraphs[-1].paragraph_id}

    containing = {
        paragraph.paragraph_id
        for paragraph in paragraphs
        if paragraph.start < position < paragraph.end
    }
    if containing:
        return containing

    # An insertion exactly at an old paragraph boundary is conservatively
    # assigned to both sides so it cannot be lost by union extraction.
    adjacent = {
        paragraph.paragraph_id
        for paragraph in paragraphs
        if paragraph.start == position or paragraph.end == position
    }
    return adjacent or {paragraphs[-1].paragraph_id}


def _derived_modified_ids(
    paragraphs: list[_SavedParagraph],
    opcodes: list[Opcode],
) -> set[int]:
    modified: set[int] = set()
    for tag, old_start, old_end, _, _ in opcodes:
        if tag == "equal":
            continue
        if old_start == old_end:
            modified.update(_paragraphs_at_insertion(paragraphs, old_start))
        else:
            modified.update(
                _paragraphs_overlapping(paragraphs, old_start, old_end)
            )
    return modified


def _contiguous_groups(paragraph_ids: set[int]) -> list[tuple[int, ...]]:
    groups: list[list[int]] = []
    for paragraph_id in sorted(paragraph_ids):
        if not groups or paragraph_id != groups[-1][-1] + 1:
            groups.append([paragraph_id])
        else:
            groups[-1].append(paragraph_id)
    return [tuple(group) for group in groups]


def _boundary_candidates(position: int, opcodes: list[Opcode]) -> list[int]:
    candidates: list[int] = []
    for tag, old_start, old_end, current_start, current_end in opcodes:
        if not old_start <= position <= old_end:
            continue
        if tag == "equal":
            candidates.append(current_start + (position - old_start))
        elif old_start == old_end:
            candidates.extend((current_start, current_end))
        elif position == old_start:
            candidates.append(current_start)
        elif position == old_end:
            candidates.append(current_end)
        else:
            candidates.extend((current_start, current_end))
    return candidates


def _current_span_for_old_span(
    old_start: int,
    old_end: int,
    opcodes: list[Opcode],
) -> tuple[int, int]:
    start_candidates = _boundary_candidates(old_start, opcodes)
    end_candidates = _boundary_candidates(old_end, opcodes)
    if not start_candidates or not end_candidates:
        raise ValueError("current_text cannot be aligned to saved paragraph boundaries")
    current_start = min(start_candidates)
    current_end = max(end_candidates)
    if current_end < current_start:
        raise ValueError("current_text paragraph alignment is inconsistent")
    return current_start, current_end


def _align_unions(
    paragraphs: list[_SavedParagraph],
    current_text: str,
    hinted_ids: set[int],
) -> list[_ParagraphUnion]:
    saved_text = "".join(paragraph.text for paragraph in paragraphs)
    opcodes = list(
        SequenceMatcher(None, saved_text, current_text, autojunk=False).get_opcodes()
    )
    marked_ids = _derived_modified_ids(paragraphs, opcodes) | hinted_ids
    if not marked_ids:
        raise ValueError("current_text changed but no modified paragraph was identified")

    paragraph_by_id = {item.paragraph_id: item for item in paragraphs}
    unions: list[_ParagraphUnion] = []
    union_by_first_id: dict[int, _ParagraphUnion] = {}
    covered_ids: set[int] = set()
    for group in _contiguous_groups(marked_ids):
        first = paragraph_by_id[group[0]]
        last = paragraph_by_id[group[-1]]
        current_start, current_end = _current_span_for_old_span(
            first.start,
            last.end,
            opcodes,
        )
        union = _ParagraphUnion(
            paragraph_ids=group,
            current_text=current_text[current_start:current_end],
        )
        unions.append(union)
        union_by_first_id[group[0]] = union
        covered_ids.update(group)

    reconstructed: list[str] = []
    for paragraph in paragraphs:
        union = union_by_first_id.get(paragraph.paragraph_id)
        if union is not None:
            reconstructed.append(union.current_text)
        elif paragraph.paragraph_id not in covered_ids:
            reconstructed.append(paragraph.text)
    if "".join(reconstructed) != current_text:
        raise ValueError(
            "current_text cannot be reconstructed from modified unions and saved text"
        )
    return unions


def _validated_processor_output(
    value: object,
    source_text: str,
    processor_name: str,
) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{processor_name} must return a list of strings")
    if "".join(value) != source_text:
        raise ValueError(f"{processor_name} must preserve its source text exactly")
    return value


def _validated_vector(value: object) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("embedder must return a sequence of numbers")
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise TypeError("embedding vector must contain only numbers")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise TypeError("embedding vector must contain only numbers") from exc
        if not math.isfinite(number):
            raise ValueError("embedding vector values must be finite")
        vector.append(number)
    if not vector:
        raise ValueError("embedding vector must not be empty")
    return tuple(vector)


def _new_chunk_uuid(
    runtime: WizardRuntime,
    unavailable_ids: set[str],
) -> str:
    value = runtime.uuid_factory()
    try:
        chunk_id = str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("uuid_factory returned an invalid chunk_id") from exc
    if chunk_id in unavailable_ids:
        raise ValueError(f"uuid_factory returned duplicate chunk_id {chunk_id!r}")
    unavailable_ids.add(chunk_id)
    return chunk_id


def _batch_embeddings(
    runtime: WizardRuntime,
    texts: list[str],
) -> list[tuple[float, ...]]:
    if not texts:
        return []
    raw_vectors = runtime.embedder.embed_many(texts)
    if isinstance(raw_vectors, (str, bytes)) or not isinstance(
        raw_vectors, Sequence
    ):
        raise TypeError("embedder must return a sequence of vectors")
    if len(raw_vectors) != len(texts):
        raise ValueError("embedder returned the wrong number of vectors")
    vectors = [_validated_vector(vector) for vector in raw_vectors]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("embedding vectors must have consistent dimensions")
    return vectors


def _recover_save(
    *,
    collection: object,
    document_map: object,
    paragraph_map: object,
    document_id: str,
    old_paragraph_data: dict[int, str],
    old_chunk_mappings: dict[int, list[str]],
    deleted_snapshots: tuple[ChunkRecord, ...],
    attempted_insert_ids: list[str],
    attempted_retained_ids: list[str],
    old_paragraph_by_chunk: dict[str, int],
) -> tuple[list[BaseException], tuple[str, ...]]:
    errors: list[BaseException] = []
    unresolved_ids: list[str] = []

    if attempted_retained_ids:
        old_ids = {
            chunk_id: old_paragraph_by_chunk[chunk_id]
            for chunk_id in attempted_retained_ids
        }
        try:
            collection.update_paragraph_ids(old_ids)
            collection.verify_paragraph_ids(old_ids)
        except Exception as exc:
            errors.append(exc)
            unresolved_ids.extend(attempted_retained_ids)
    if attempted_insert_ids:
        try:
            collection.delete_chunks(list(dict.fromkeys(attempted_insert_ids)))
        except Exception as exc:
            errors.append(exc)
            unresolved_ids.extend(attempted_insert_ids)
    if deleted_snapshots:
        try:
            collection.restore_chunks(deleted_snapshots)
        except Exception as exc:
            errors.append(exc)
            unresolved_ids.extend(record.chunk_id for record in deleted_snapshots)
    try:
        paragraph_map.replace_document(document_id, old_chunk_mappings)
    except Exception as exc:
        errors.append(exc)
    try:
        document_map.restore_document(document_id, old_paragraph_data)
    except Exception as exc:
        errors.append(exc)

    return errors, tuple(dict.fromkeys(unresolved_ids))


def save_wizard(
    user_id: str,
    document_id: str,
    collection_type: str,
    current_text: str,
    modified_paragraph_ids: list[int] | None = None,
    *,
    runtime: WizardRuntime | None = None,
) -> None:
    """Execute the documented destructive eight-step wizard save pipeline."""

    if not isinstance(current_text, str):
        raise TypeError("current_text must be a string")
    active_runtime = resolve_runtime(runtime)
    document_map = active_runtime.document_map(user_id, collection_type)
    paragraph_map = active_runtime.paragraph_map(user_id, collection_type)
    paragraph_data = document_map.get_paragraph_data(document_id)
    old_paragraphs = _saved_paragraphs(paragraph_data)
    old_paragraph_ids = {item.paragraph_id for item in old_paragraphs}
    hinted_ids = _validated_modified_ids(
        modified_paragraph_ids,
        old_paragraph_ids,
    )
    saved_text = "".join(item.text for item in old_paragraphs)
    if current_text == saved_text:
        return

    # Steps 1-2: identify and align every contiguous modified union before
    # making the first destructive call.
    unions = _align_unions(old_paragraphs, current_text, hinted_ids)
    old_chunks_by_paragraph = paragraph_map.get_document_chunks(document_id)
    existing_chunk_ids = {
        chunk_id
        for chunk_ids in old_chunks_by_paragraph.values()
        for chunk_id in chunk_ids
    }
    collection = active_runtime.collection(user_id, collection_type)

    targeted_ids = {
        chunk_id
        for union in unions
        for paragraph_id in union.paragraph_ids
        for chunk_id in old_chunks_by_paragraph.get(paragraph_id, [])
    }
    snapshots: list[ChunkRecord] = []
    try:
        for union in unions:
            snapshots.extend(
                collection.snapshot_by_paragraphs(
                    document_id, list(union.paragraph_ids)
                )
            )
        if {record.chunk_id for record in snapshots} != targeted_ids:
            raise RuntimeError("storage snapshot does not match paragraph mappings")
    except Exception as exc:
        raise WizardSaveError(
            "step_3_snapshot", affected_chunk_ids=sorted(targeted_ids)
        ) from exc

    deleted_snapshots = tuple(snapshots)
    attempted_insert_ids: list[str] = []
    inserted_ids: list[str] = []
    attempted_retained_ids: list[str] = []
    updated_retained_ids: list[str] = []
    old_paragraph_by_chunk = {
        chunk_id: paragraph_id
        for paragraph_id, chunk_ids in old_chunks_by_paragraph.items()
        for chunk_id in chunk_ids
    }
    failed_stage = "step_3"
    try:
        # Step 3: all verified deletions complete before semantic processing.
        for union in unions:
            collection.delete_by_paragraphs(document_id, list(union.paragraph_ids))

        # Steps 4-5: losslessly split/chunk, then embed the whole save batch.
        failed_stage = "step_4"
        staged_by_union: dict[int, list[_ParagraphNode]] = {}
        unavailable_chunk_ids = set(existing_chunk_ids)
        pending_chunks: list[_StagedChunk] = []
        for union in unions:
            split_paragraphs = _validated_processor_output(
                active_runtime.paragraph_splitter(union.current_text),
                union.current_text,
                "paragraph_splitter",
            )
            staged_paragraphs: list[_ParagraphNode] = []
            for paragraph_text in split_paragraphs:
                chunks = _validated_processor_output(
                    active_runtime.paragraph_chunker(paragraph_text),
                    paragraph_text,
                    "paragraph_chunker",
                )
                staged_chunks: list[_StagedChunk] = []
                for chunk_text in chunks:
                    if not chunk_text.strip():
                        raise ValueError("paragraph_chunker returned an empty chunk")
                    chunk = _StagedChunk(
                        chunk_id=_new_chunk_uuid(
                            active_runtime, unavailable_chunk_ids
                        ),
                        raw_text=chunk_text,
                        vector=(),
                    )
                    staged_chunks.append(chunk)
                    pending_chunks.append(chunk)
                staged_paragraphs.append(
                    _ParagraphNode(
                        raw_text=paragraph_text,
                        chunks=tuple(staged_chunks),
                        old_paragraph_id=None,
                    )
                )
            staged_by_union[union.paragraph_ids[0]] = staged_paragraphs

        failed_stage = "step_5"
        vectors = _batch_embeddings(
            active_runtime,
            [chunk.raw_text for chunk in pending_chunks],
        )
        vectors_by_id = {
            chunk.chunk_id: vector
            for chunk, vector in zip(pending_chunks, vectors, strict=True)
        }
        staged_by_union = {
            first_id: [
                _ParagraphNode(
                    raw_text=node.raw_text,
                    chunks=tuple(
                        _StagedChunk(
                            chunk.chunk_id,
                            chunk.raw_text,
                            vectors_by_id[chunk.chunk_id],
                        )
                        for chunk in node.chunks
                    ),
                    old_paragraph_id=None,
                )
                for node in nodes
            ]
            for first_id, nodes in staged_by_union.items()
        }

        # Step 6: substitute unions and renumber the whole document.
        failed_stage = "step_6"
        union_ids = {
            paragraph_id for union in unions for paragraph_id in union.paragraph_ids
        }
        nodes: list[_ParagraphNode] = []
        for old_paragraph in old_paragraphs:
            if old_paragraph.paragraph_id in staged_by_union:
                nodes.extend(staged_by_union[old_paragraph.paragraph_id])
            elif old_paragraph.paragraph_id not in union_ids:
                retained_chunks = tuple(
                    _StagedChunk(chunk_id, "", ())
                    for chunk_id in old_chunks_by_paragraph.get(
                        old_paragraph.paragraph_id, []
                    )
                )
                nodes.append(
                    _ParagraphNode(
                        raw_text=old_paragraph.text,
                        chunks=retained_chunks,
                        old_paragraph_id=old_paragraph.paragraph_id,
                    )
                )
        if not nodes:
            nodes = [_ParagraphNode(raw_text="", chunks=(), old_paragraph_id=None)]
        if "".join(node.raw_text for node in nodes) != current_text:
            raise RuntimeError("renumbered paragraphs do not reconstruct current_text")

        final_paragraph_data: dict[int, str] = {}
        final_chunk_mappings: dict[int, list[str]] = {}
        unmodified_updates: dict[str, int] = {}
        new_chunks: list[tuple[int, _StagedChunk]] = []
        for final_paragraph_id, node in enumerate(nodes, start=1):
            final_paragraph_data[final_paragraph_id] = node.raw_text
            final_chunk_mappings[final_paragraph_id] = [
                chunk.chunk_id for chunk in node.chunks
            ]
            if node.old_paragraph_id is None:
                new_chunks.extend(
                    (final_paragraph_id, chunk) for chunk in node.chunks
                )
            elif node.old_paragraph_id != final_paragraph_id:
                for chunk in node.chunks:
                    unmodified_updates[chunk.chunk_id] = final_paragraph_id

        # Prevalidate both halves of Step 8 before the first Step 7 write.
        paragraph_map.validate_document_replacement(document_id, final_chunk_mappings)
        document_map.validate_update(document_id, final_paragraph_data)

        failed_stage = "step_7a"
        for final_paragraph_id, chunk in new_chunks:
            attempted_insert_ids.append(chunk.chunk_id)
            collection.insert_chunk(
                document_id,
                final_paragraph_id,
                chunk.chunk_id,
                chunk.raw_text,
                chunk.vector,
            )
            inserted_ids.append(chunk.chunk_id)

        failed_stage = "step_7b"
        if unmodified_updates:
            attempted_retained_ids.extend(unmodified_updates)
            try:
                collection.update_paragraph_ids(unmodified_updates)
                updated_retained_ids.extend(unmodified_updates)
                collection.verify_paragraph_ids(unmodified_updates)
            except PartialParagraphUpdateError as exc:
                updated_retained_ids.extend(exc.completed_chunk_ids)
                raise

        failed_stage = "step_8"
        paragraph_map.replace_document(document_id, final_chunk_mappings)
        document_map.update_paragraphs(document_id, final_paragraph_data)
    except Exception as exc:
        recovery_errors, unresolved = _recover_save(
            collection=collection,
            document_map=document_map,
            paragraph_map=paragraph_map,
            document_id=document_id,
            old_paragraph_data=paragraph_data,
            old_chunk_mappings=old_chunks_by_paragraph,
            deleted_snapshots=deleted_snapshots,
            attempted_insert_ids=attempted_insert_ids,
            attempted_retained_ids=attempted_retained_ids,
            old_paragraph_by_chunk=old_paragraph_by_chunk,
        )
        error_kwargs = {
            "inserted_chunk_ids": inserted_ids,
            "updated_chunk_ids": updated_retained_ids,
            "affected_chunk_ids": [record.chunk_id for record in deleted_snapshots],
        }
        if recovery_errors:
            raise WizardSaveRecoveryError(
                failed_stage,
                unresolved_chunk_ids=unresolved,
                recovery_errors=recovery_errors,
                **error_kwargs,
            ) from exc
        raise WizardSaveError(failed_stage, **error_kwargs) from exc
