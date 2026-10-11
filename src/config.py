from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    target_tokens: int = 400  # soft goal for a chunk (incl. context header)
    max_tokens: int = 512  # hard ceiling for anything splittable (incl. header)
    min_tokens: int = 100  # smaller chunks are merged with neighbours
    overlap_tokens: int = 50  # ~12% of target; whole sentences only
    tokenizer: str = "cl100k_base"
    table_strategy: str = "lines"  # PyMuPDF find_tables: lines | lines_strict | text
    keep_headers_footers: bool = False
    ocr: bool = False  # OCR pages that have no text layer (needs pytesseract)


def make_token_counter(name: str) -> tuple[Callable[[str], int], str]:
    """tiktoken if installed (and its vocab is available), else a conservative estimate."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding(name)
        return (
            lambda s: len(enc.encode(s, disallowed_special=())) if s else 0
        ), f"tiktoken:{name}"
    except Exception:  # not installed / offline

        def approx(s: str) -> int:
            return int(max(len(s) / 4.0, len(s.split()) * 1.3)) if s else 0

        return approx, "approx(max(chars/4, words*1.3))"
