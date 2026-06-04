import hashlib
import re
import unicodedata


def normalize_str(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", value).strip().lower()


def normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    # Nigerian fallback: strip leading 234 / 0
    if digits.startswith("234") and len(digits) > 10:
        digits = digits[3:]
    if digits.startswith("0"):
        digits = digits[1:]
    return digits


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
