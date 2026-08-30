"""Empty-state-only Weaviate vector-profile maintenance migration."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re

from weaviate.classes.config import Configure

from backend.config import get_collection_name
from backend.model_config import EMBEDDING_MODEL, EMBEDDING_VECTOR_PROFILE
from backend.weaviate_client.client import (
    WeaviateManager,
    _CHUNK_CONTRACT,
    _CONVERSATION_CONTRACT,
    _chunk_properties,
    _collection_description,
    _conversation_properties,
    _validate_properties,
    _validate_vector_config,
)


_MANIFEST_NAME = "vector-migration-manifest.json"
_MANIFEST_VERSION = "2.0"
_MIGRATION_SCOPE = "disposable-empty-prototype-state"
_COLLECTION_PATTERN = re.compile(
    r"^RagUser_(?P<token>[A-Z2-7]+)_(?P<suffix>Conversations|KnowledgeFacts|Policy)$"
)
_SUFFIX_TYPES = {
    "Conversations": "conversations",
    "KnowledgeFacts": "knowledge_facts",
    "Policy": "policy",
}
_REQUIRED_TYPES = frozenset(_SUFFIX_TYPES.values())


class VectorMigrationError(RuntimeError):
    """Raised when empty-state migration safety cannot be proven."""


def _decoded_collection(name: str) -> tuple[str, str] | None:
    match = _COLLECTION_PATTERN.fullmatch(name)
    if match is None:
        return None
    token = match.group("token")
    padding = "=" * ((8 - len(token) % 8) % 8)
    try:
        user_id = base64.b32decode(token + padding).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise VectorMigrationError(f"collection {name!r} has an invalid user token") from exc
    collection_type = _SUFFIX_TYPES[match.group("suffix")]
    if get_collection_name(user_id, collection_type) != name:
        raise VectorMigrationError(f"collection {name!r} is not canonically encoded")
    return user_id, collection_type


def _validate_source_schema(collection: object, name: str, collection_type: str) -> None:
    config = collection.config.get()
    configured_name = getattr(config, "name", None)
    if configured_name != name:
        raise VectorMigrationError(f"collection {name!r} returned the wrong schema")
    references = getattr(config, "references", None)
    if references not in (None, []):
        raise VectorMigrationError(f"collection {name!r} contains references")
    contract = (
        _CONVERSATION_CONTRACT
        if collection_type == "conversations"
        else _CHUNK_CONTRACT
    )
    try:
        _validate_properties(name, config, contract)
        _validate_vector_config(name, config)
    except Exception as exc:
        raise VectorMigrationError(f"collection {name!r} has an incompatible schema") from exc


def _total_count(collection: object, name: str) -> int:
    result = collection.aggregate.over_all(total_count=True)
    count = getattr(result, "total_count", None)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise VectorMigrationError(f"collection {name!r} returned a malformed object count")
    return count


def _canonical_entries(collections: object) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    grouped: dict[str, set[str]] = {}
    names = collections.list_all(simple=True)
    if not isinstance(names, Mapping) and (
        isinstance(names, (str, bytes)) or not isinstance(names, Sequence)
    ):
        raise VectorMigrationError("Weaviate returned a malformed collection list")
    for name in sorted(names):
        if not isinstance(name, str):
            raise VectorMigrationError("Weaviate returned a malformed collection name")
        decoded = _decoded_collection(name)
        if decoded is None:
            continue
        user_id, collection_type = decoded
        grouped.setdefault(user_id, set()).add(collection_type)
        entries.append(
            {
                "name": name,
                "user_id": user_id,
                "collection_type": collection_type,
                "count": 0,
            }
        )
    if not grouped or any(types != _REQUIRED_TYPES for types in grouped.values()):
        raise VectorMigrationError(
            "every user must have exactly the three canonical collections"
        )
    return entries


def _require_disposable_confirmation(confirmed: bool) -> None:
    if confirmed is not True:
        raise VectorMigrationError(
            "migration requires explicit confirmation that process-local state is disposable"
        )


def _require_existing_targets_empty(
    collections: object,
    entries: Sequence[Mapping[str, object]],
) -> None:
    for entry in entries:
        name = str(entry["name"])
        if not collections.exists(name):
            continue
        collection = collections.use(name)
        _validate_source_schema(collection, name, str(entry["collection_type"]))
        count = _total_count(collection, name)
        if count != 0:
            raise VectorMigrationError(
                f"collection {name!r} contains {count} objects; populated state is unsupported"
            )


def export_collections(
    manager: WeaviateManager,
    export_directory: Path,
    *,
    confirm_disposable_process_state: bool = False,
) -> Path:
    """Record an empty-state migration manifest without preserving mappings."""

    if not isinstance(manager, WeaviateManager):
        raise TypeError("manager must be a WeaviateManager")
    _require_disposable_confirmation(confirm_disposable_process_state)
    root = export_directory.expanduser().resolve()
    manifest_path = root / _MANIFEST_NAME
    if manifest_path.exists():
        raise VectorMigrationError("migration export already exists")

    collections = manager.client.collections
    entries = _canonical_entries(collections)
    _require_existing_targets_empty(collections, entries)

    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    manifest = {
        "schema_version": _MANIFEST_VERSION,
        "migration_scope": _MIGRATION_SCOPE,
        "mapping_state_preserved": False,
        "embedding_model": EMBEDDING_MODEL,
        "vector_profile": EMBEDDING_VECTOR_PROFILE,
        "collections": entries,
    }
    with manifest_path.open("x", encoding="utf-8") as output:
        os.chmod(manifest_path, 0o600)
        output.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    return manifest_path


def _load_manifest(export_directory: Path) -> list[dict[str, object]]:
    manifest_path = export_directory.expanduser().resolve() / _MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VectorMigrationError("migration manifest is missing or malformed") from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != _MANIFEST_VERSION
        or manifest.get("migration_scope") != _MIGRATION_SCOPE
        or manifest.get("mapping_state_preserved") is not False
        or manifest.get("embedding_model") != EMBEDDING_MODEL
        or manifest.get("vector_profile") != EMBEDDING_VECTOR_PROFILE
    ):
        raise VectorMigrationError("migration manifest is not an empty-state manifest")
    raw_entries = manifest.get("collections")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise VectorMigrationError("migration manifest collections are malformed")

    entries: list[dict[str, object]] = []
    names: set[str] = set()
    grouped: dict[str, set[str]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise VectorMigrationError("migration manifest collection entry is invalid")
        name = raw_entry.get("name")
        user_id = raw_entry.get("user_id")
        collection_type = raw_entry.get("collection_type")
        count = raw_entry.get("count")
        if (
            set(raw_entry) != {"name", "user_id", "collection_type", "count"}
            or not isinstance(name, str)
            or name in names
            or not isinstance(user_id, str)
            or not isinstance(collection_type, str)
            or isinstance(count, bool)
            or count != 0
        ):
            raise VectorMigrationError("migration manifest collection entry is invalid")
        try:
            canonical_name = get_collection_name(user_id, collection_type)
        except (TypeError, ValueError) as exc:
            raise VectorMigrationError("migration manifest collection entry is invalid") from exc
        if canonical_name != name:
            raise VectorMigrationError("migration manifest collection entry is invalid")
        names.add(name)
        grouped.setdefault(user_id, set()).add(collection_type)
        entries.append(dict(raw_entry))
    if any(types != _REQUIRED_TYPES for types in grouped.values()):
        raise VectorMigrationError("migration manifest has incomplete user collections")
    return entries


def _require_no_new_canonical_collections(
    collections: object,
    entries: Sequence[Mapping[str, object]],
) -> None:
    manifest_names = {str(entry["name"]) for entry in entries}
    current_names = {
        name
        for name in collections.list_all(simple=True)
        if isinstance(name, str) and _decoded_collection(name) is not None
    }
    if current_names - manifest_names:
        raise VectorMigrationError("canonical collection set changed after export")


def rebuild_collections(
    manager: WeaviateManager,
    export_directory: Path,
    *,
    confirm_disposable_process_state: bool = False,
) -> None:
    """Recreate only confirmed-empty canonical collections under the new profile."""

    if not isinstance(manager, WeaviateManager):
        raise TypeError("manager must be a WeaviateManager")
    _require_disposable_confirmation(confirm_disposable_process_state)
    entries = _load_manifest(export_directory)
    collections = manager.client.collections
    _require_no_new_canonical_collections(collections, entries)
    _require_existing_targets_empty(collections, entries)

    for entry in entries:
        name = str(entry["name"])
        if collections.exists(name):
            collections.use(name).config.update(
                description=_collection_description("rebuilding")
            )

    rebuilt: list[object] = []
    for entry in entries:
        name = str(entry["name"])
        collection_type = str(entry["collection_type"])
        if collections.exists(name):
            collections.delete(name)
        collection = collections.create(
            name=name,
            description=_collection_description("rebuilding"),
            properties=(
                _conversation_properties()
                if collection_type == "conversations"
                else _chunk_properties()
            ),
            vector_config=Configure.Vectors.self_provided(),
        )
        _validate_source_schema(collection, name, collection_type)
        if _total_count(collection, name) != 0:
            raise VectorMigrationError("a rebuilt empty collection unexpectedly contains objects")
        rebuilt.append(collection)

    for collection in rebuilt:
        collection.config.update(description=_collection_description())


def verify_collections(manager: WeaviateManager, export_directory: Path) -> None:
    """Verify ready schemas and zero objects without mutating storage."""

    if not isinstance(manager, WeaviateManager):
        raise TypeError("manager must be a WeaviateManager")
    entries = _load_manifest(export_directory)
    collections = manager.client.collections
    current_names = {
        name
        for name in collections.list_all(simple=True)
        if isinstance(name, str) and _decoded_collection(name) is not None
    }
    if current_names != {str(entry["name"]) for entry in entries}:
        raise VectorMigrationError("canonical collection set does not match the manifest")
    for entry in entries:
        name = str(entry["name"])
        collection = collections.use(name)
        config = collection.config.get()
        if getattr(config, "description", None) != _collection_description():
            raise VectorMigrationError(f"collection {name!r} is not ready")
        _validate_source_schema(collection, name, str(entry["collection_type"]))
        if _total_count(collection, name) != 0:
            raise VectorMigrationError(f"collection {name!r} is no longer empty")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate empty canonical Weaviate collections during maintenance"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("directory", type=Path)
    export_parser.add_argument("--confirm-disposable-process-state", action="store_true")
    rebuild_parser = subparsers.add_parser("rebuild")
    rebuild_parser.add_argument("directory", type=Path)
    rebuild_parser.add_argument("--confirm-maintenance", action="store_true")
    rebuild_parser.add_argument("--confirm-disposable-process-state", action="store_true")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()

    manager = WeaviateManager()
    manager.connect()
    try:
        if arguments.command == "export":
            export_collections(
                manager,
                arguments.directory,
                confirm_disposable_process_state=arguments.confirm_disposable_process_state,
            )
        elif arguments.command == "rebuild":
            if not arguments.confirm_maintenance:
                parser.error("rebuild requires --confirm-maintenance")
            rebuild_collections(
                manager,
                arguments.directory,
                confirm_disposable_process_state=arguments.confirm_disposable_process_state,
            )
        else:
            verify_collections(manager, arguments.directory)
    finally:
        manager.disconnect()


if __name__ == "__main__":
    main()


__all__ = [
    "VectorMigrationError",
    "export_collections",
    "rebuild_collections",
    "verify_collections",
]
