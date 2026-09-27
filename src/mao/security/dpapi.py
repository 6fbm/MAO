"""Minimal Windows DPAPI wrapper (CryptProtectData / CryptUnprotectData).

Data encrypted this way can only be decrypted by the same Windows user on
the same machine. No third-party dependency is required.
"""

from __future__ import annotations

import sys

CRYPTPROTECT_UI_FORBIDDEN = 0x01


def available() -> bool:
    return sys.platform == "win32"


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _DataBlob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    _crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    _crypt32.CryptProtectData.restype = wintypes.BOOL
    _crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    _crypt32.CryptUnprotectData.restype = wintypes.BOOL
    _kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel32.LocalFree.restype = ctypes.c_void_p

    def _make_blob(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
        buffer = ctypes.create_string_buffer(data, len(data))
        blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
        return blob, buffer

    def protect(data: bytes, entropy: bytes = b"") -> bytes:
        in_blob, _in_buf = _make_blob(data)
        entropy_blob, _ent_buf = _make_blob(entropy) if entropy else (None, None)
        out_blob = _DataBlob()
        ok = _crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            "mao-secrets",
            ctypes.byref(entropy_blob) if entropy_blob is not None else None,
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            _kernel32.LocalFree(ctypes.cast(out_blob.pbData, ctypes.c_void_p))

    def unprotect(data: bytes, entropy: bytes = b"") -> bytes:
        in_blob, _in_buf = _make_blob(data)
        entropy_blob, _ent_buf = _make_blob(entropy) if entropy else (None, None)
        out_blob = _DataBlob()
        ok = _crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            ctypes.byref(entropy_blob) if entropy_blob is not None else None,
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            _kernel32.LocalFree(ctypes.cast(out_blob.pbData, ctypes.c_void_p))

else:  # pragma: no cover - non-Windows

    def protect(data: bytes, entropy: bytes = b"") -> bytes:
        raise OSError("DPAPI is only available on Windows")

    def unprotect(data: bytes, entropy: bytes = b"") -> bytes:
        raise OSError("DPAPI is only available on Windows")
