from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

MessageId = int


@dataclass
class ImmutableStore:
    _messages: list[dict[str, Any]] = field(default_factory=list)

    def append(self, message: dict[str, Any]) -> MessageId:
        msg_id = len(self._messages)
        self._messages.append(message)
        return msg_id

    def get(self, msg_id: MessageId) -> dict[str, Any] | None:
        if 0 <= msg_id < len(self._messages):
            return self._messages[msg_id]
        return None

    def get_many(
        self, msg_ids: list[MessageId]
    ) -> list[tuple[MessageId, dict[str, Any]]]:
        return [
            (mid, self._messages[mid])
            for mid in msg_ids
            if 0 <= mid < len(self._messages)
        ]

    def __len__(self) -> int:
        return len(self._messages)
