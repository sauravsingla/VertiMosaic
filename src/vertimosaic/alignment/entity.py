from __future__ import annotations

import hashlib


class EntityAligner:
    """Research pseudonymization interface; this is not a PSI protocol."""

    def __init__(self, salt: str) -> None:
        if not salt:
            raise ValueError("salt must not be empty")
        self.salt = salt

    def pseudonymize(self, source_id: str) -> str:
        payload = f"{self.salt}:{source_id}".encode()
        return hashlib.sha256(payload).hexdigest()
