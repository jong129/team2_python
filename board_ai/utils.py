import base64
import binascii
from fastapi import HTTPException

def cut_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars and max_chars > 0 and len(s) > max_chars:
        return s[:max_chars]
    return s

def decode_base64_image(image_b64: str) -> bytes:
    b64 = (image_b64 or "").strip()
    if b64.startswith("data:") and "base64," in b64:
        b64 = b64.split("base64,", 1)[1].strip()

    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="imageBase64 decode failed")

def make_data_url(image_bytes: bytes, content_type: str) -> str:
    ct = (content_type or "").strip().lower()
    if not ct.startswith("image/"):
        ct = "image/png"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{ct};base64,{b64}"
