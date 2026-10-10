import base64
import re
import zlib


def pdf_text(data: bytes) -> bytes:
    """Decodes ReportLab's ASCII85 + Flate content streams so tests can search the text."""
    out = b""
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        raw = m.group(1).strip()
        if raw.endswith(b"~>"):
            raw = raw[:-2]
        try:
            out += zlib.decompress(base64.a85decode(raw, ignorechars=b" \t\n\r\x0b"))
        except Exception:
            pass
    # Undo PDF string escaping of parentheses and backslashes
    return out.replace(b"\\(", b"(").replace(b"\\)", b")").replace(b"\\\\", b"\\")
