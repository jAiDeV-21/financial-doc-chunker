from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from chunker import Chunker
from config import ChunkConfig, make_token_counter
from data_models import Chunk
from extraction import extract
from structure import build_sections

log = logging.getLogger("doc_chunker")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "doc"


def process_document(path: Path, cfg: ChunkConfig) -> tuple[list[Chunk], dict, dict]:
    count, counter_name = make_token_counter(cfg.tokenizer)
    parsed = extract(path, cfg)
    doc_id = f"{_slug(path.stem)}-{hashlib.sha1(path.read_bytes()).hexdigest()[:8]}"
    chunker = Chunker(cfg, count)
    chunks = chunker.chunk_document(parsed, doc_id, path.name, counter_name)
    outline = [
        {
            "path": s.path,
            "level": s.level,
            "page": s.page,
            "tokens": chunker._block_tokens(s.blocks),
        }
        for s in build_sections(parsed.blocks)
    ]
    document = {
        "doc_id": doc_id,
        "title": parsed.title,
        "source_file": path.name,
        "n_pages": parsed.n_pages,
        "n_chunks": len(chunks),
        "token_counter": counter_name,
        "config": asdict(cfg),
        "extraction": parsed.info,
    }
    return chunks, outline, document


def write_outputs(
    chunks: list[Chunk], outline: list, document: dict, out_dir: Path, fmt: str
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(document["source_file"]).stem
    written = []
    if fmt == "jsonl":
        p = out_dir / f"{stem}.chunks.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    else:
        p = out_dir / f"{stem}.chunks.json"
        p.write_text(
            json.dumps(
                {"document": document, "chunks": [asdict(c) for c in chunks]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    written.append(p)
    o = out_dir / f"{stem}.outline.json"
    o.write_text(
        json.dumps(
            {"document": document, "sections": outline}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    return written + [o]


def summarize(chunks: list[Chunk], cfg: ChunkConfig) -> str:
    if not chunks:
        return "  no chunks produced"
    toks = sorted(c.token_count for c in chunks)
    pct = lambda p: toks[int(p * (len(toks) - 1))]  # noqa: E731
    by_type = Counter(c.content_type for c in chunks)
    prose = [c for c in chunks if c.content_type != "table"]
    clean_end = sum(bool(re.search(r"[.!?:;)\"”\]]\s*$", c.text)) for c in prose)
    over = sum(c.token_count > cfg.max_tokens for c in chunks)
    return (
        f"  chunks={len(chunks)}  types={dict(by_type)}\n"
        f"  tokens: min={toks[0]} p50={pct(.5)} p90={pct(.9)} max={toks[-1]}  "
        f"(> max_tokens: {over})\n"
        f"  non-table chunks ending on sentence punctuation: {clean_end}/{len(prose)}\n"
        f"  chunks with overlap: {sum(c.overlap_prefix_chars > 0 for c in chunks)}  "
        f"flagged: {dict(Counter(f for c in chunks for f in c.flags))}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Structure-aware document chunker for RAG")
    ap.add_argument(
        "inputs", nargs="+", help="PDF / DOCX / TXT / MD files or directories"
    )
    ap.add_argument("-o", "--out", default="chunks_out", help="output directory")
    ap.add_argument("--format", choices=["jsonl", "json"], default="jsonl")
    d = ChunkConfig()
    ap.add_argument("--target-tokens", type=int, default=d.target_tokens)
    ap.add_argument("--max-tokens", type=int, default=d.max_tokens)
    ap.add_argument("--min-tokens", type=int, default=d.min_tokens)
    ap.add_argument("--overlap-tokens", type=int, default=d.overlap_tokens)
    ap.add_argument("--tokenizer", default=d.tokenizer, help="tiktoken encoding name")
    ap.add_argument(
        "--table-strategy",
        choices=["lines", "lines_strict", "text"],
        default=d.table_strategy,
        help="PyMuPDF table detection; use 'text' for tables without ruling lines",
    )
    ap.add_argument("--keep-headers-footers", action="store_true")
    ap.add_argument("--ocr", action="store_true", help="OCR pages with no text layer")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO, format="%(message)s"
    )

    cfg = ChunkConfig(
        a.target_tokens,
        a.max_tokens,
        a.min_tokens,
        a.overlap_tokens,
        a.tokenizer,
        a.table_strategy,
        a.keep_headers_footers,
        a.ocr,
    )
    if not (
        0 < cfg.min_tokens < cfg.target_tokens <= cfg.max_tokens
        and cfg.overlap_tokens < cfg.target_tokens
    ):
        ap.error("need 0 < min < target <= max and overlap < target")

    files: list[Path] = []
    for inp in a.inputs:
        p = Path(inp)
        files += (
            sorted(
                x
                for x in p.rglob("*")
                if x.suffix.lower() in {".pdf", ".docx", ".txt", ".md", ".markdown"}
            )
            if p.is_dir()
            else [p]
        )
    rc = 0
    for f in files:
        try:
            chunks, outline, document = process_document(f, cfg)
            paths = write_outputs(chunks, outline, document, Path(a.out), a.format)
            log.info("%s -> %s", f.name, ", ".join(str(p) for p in paths))
            log.info(
                "  tokenizer: %s\n%s", document["token_counter"], summarize(chunks, cfg)
            )
        except Exception as e:  # noqa: BLE001
            log.error("FAILED %s: %s", f, e, exc_info=a.verbose)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
