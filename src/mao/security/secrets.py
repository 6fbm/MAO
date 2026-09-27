"""Local secret store for API keys.

Windows: the JSON payload is encrypted with DPAPI (bound to the Windows
user). Other platforms: a plain file with owner-only permissions (clearly
reported as weaker). Keys are never printed – only short fingerprints.
Environment variables remain the alternative for users who prefer them.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path

from mao.core.errors import ConfigError
from mao.security import dpapi
from mao.security.redaction import Redactor

_HEADER_DPAPI = b"MAO-DPAPI1\n"
_HEADER_PLAIN = b"MAO-PLAIN1\n"
_ENTROPY = b"multi-ai-orchestrator/secret-store/v1"


def fingerprint(secret: str) -> str:
    return "#" + hashlib.sha256(secret.encode("utf-8")).hexdigest()[:8]


class SecretStore:
    def __init__(self, path: Path, redactor: Redactor) -> None:
        self.path = path
        self._redactor = redactor
        self._lock = threading.Lock()
        self._data: dict[str, list[str]] | None = None

    @property
    def backend(self) -> str:
        return "Windows DPAPI (encrypted per user)" if dpapi.available() else "File with owner-only permissions (unencrypted)"

    # ------------------------------------------------------------------ io

    def _load(self) -> dict[str, list[str]]:
        if self._data is not None:
            return self._data
        data: dict[str, list[str]] = {}
        if self.path.exists():
            raw = self.path.read_bytes()
            try:
                if raw.startswith(_HEADER_DPAPI):
                    payload = dpapi.unprotect(raw[len(_HEADER_DPAPI) :], _ENTROPY)
                elif raw.startswith(_HEADER_PLAIN):
                    payload = raw[len(_HEADER_PLAIN) :]
                else:
                    raise ConfigError(f"Unknown format of the secret file: {self.path}")
                parsed = json.loads(payload.decode("utf-8"))
            except ConfigError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ConfigError(
                    f"The secret file could not be decrypted ({type(exc).__name__}). "
                    "It may belong to a different Windows user."
                ) from exc
            for provider, keys in parsed.get("providers", {}).items():
                data[provider] = [k for k in keys if isinstance(k, str) and k]
        for keys in data.values():
            for key in keys:
                self._redactor.add_secret(key)
        self._data = data
        return data

    def _save(self, data: dict[str, list[str]]) -> None:
        payload = json.dumps({"version": 1, "providers": data}).encode("utf-8")
        if dpapi.available():
            blob = _HEADER_DPAPI + dpapi.protect(payload, _ENTROPY)
        else:
            blob = _HEADER_PLAIN + payload
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_bytes(blob)
        if sys.platform != "win32":
            os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ api

    def keys(self, provider: str) -> list[str]:
        with self._lock:
            return list(self._load().get(provider, []))

    def providers(self) -> list[str]:
        with self._lock:
            return sorted(p for p, keys in self._load().items() if keys)

    def fingerprints(self, provider: str) -> list[str]:
        return [fingerprint(k) for k in self.keys(provider)]

    def add(self, provider: str, key: str) -> str:
        key = key.strip()
        if len(key) < 8:
            raise ConfigError("The API key is too short.")
        with self._lock:
            data = self._load()
            keys = data.setdefault(provider, [])
            if key not in keys:
                keys.append(key)
                self._save(data)
        self._redactor.add_secret(key)
        return fingerprint(key)

    def remove(self, provider: str, ref: str) -> bool:
        """Remove by fingerprint (``#abcd1234``) or 1-based index."""
        with self._lock:
            data = self._load()
            keys = data.get(provider, [])
            target: str | None = None
            if ref.isdigit():
                index = int(ref) - 1
                if 0 <= index < len(keys):
                    target = keys[index]
            else:
                wanted = ref if ref.startswith("#") else "#" + ref
                target = next((k for k in keys if fingerprint(k) == wanted), None)
            if target is None:
                return False
            keys.remove(target)
            self._save(data)
            return True
