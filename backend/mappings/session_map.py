"""In-memory session-to-conversation mapping."""

from __future__ import annotations

from backend.mappings._common import (
    required_identifier,
    required_uuid,
    validated_user_id,
)


class SessionMap:
    """Track ordered conversation IDs for one user's sessions."""

    def __init__(self, user_id: str) -> None:
        self.user_id = validated_user_id(user_id)
        self._sessions: dict[str, list[str]] = {}
        self._conversation_owners: dict[str, str] = {}

    def create_session(self, session_id: str) -> None:
        key = required_identifier(session_id, "session_id")
        if key in self._sessions:
            raise ValueError(f"session {key!r} already exists")
        self._sessions[key] = []

    def add_conversation(self, session_id: str, conversation_id: str) -> None:
        key = required_identifier(session_id, "session_id")
        child_id = required_uuid(conversation_id, "conversation_id")
        conversations = self._sessions[key]
        owner = self._conversation_owners.get(child_id)
        if owner is not None and owner != key:
            raise ValueError(
                f"conversation {child_id!r} already belongs to session {owner!r}"
            )
        if owner is None:
            conversations.append(child_id)
            self._conversation_owners[child_id] = key

    def get_conversations(self, session_id: str) -> list[str]:
        key = required_identifier(session_id, "session_id")
        return list(self._sessions[key])

    def delete_session(self, session_id: str) -> list[str]:
        key = required_identifier(session_id, "session_id")
        conversations = self._sessions.pop(key)
        for conversation_id in conversations:
            del self._conversation_owners[conversation_id]
        return list(conversations)

    def list_sessions(self, user_id: str) -> list[str]:
        requested_user = validated_user_id(user_id)
        if requested_user != self.user_id:
            raise ValueError("user_id does not match this SessionMap")
        return list(self._sessions)
