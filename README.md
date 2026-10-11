# doc_chunker: structure-aware chunking for financial documents

```bash
pip install -r requirements.txt
python doc_chunker.py annual_report.pdf loan_agreement.docx -o out/        # JSONL (default)
python doc_chunker.py report.pdf -o out/ --format json                     # JSON with doc-level info
python doc_chunker.py report.pdf --target-tokens 300 --max-tokens 400      # for 512-token encoders w/ headroom
python doc_chunker.py statements.pdf --table-strategy text                 # tables without ruling lines
python doc_chunker.py scan.pdf --ocr                                       # image-only pages
```

Outputs per document: `<name>.chunks.jsonl` (one chunk per line) and `<name>.outline.json` (detected section tree).
Python API: `process_document(Path, ChunkConfig) -> (chunks, outline, document_info)`.

## Pipeline

| Stage | What happens |
|---|---|
| **Extract** | PDF via PyMuPDF; DOCX via python-docx; TXT/MD. Every file becomes an ordered list of typed blocks: heading, paragraph, list, table, footnote. |
| **PDF clean-up** | Running headers/footers and page numbers removed (text that repeats in the page margins on >=30% of pages); ligatures and soft hyphens normalised; end-of-line hyphenation repaired; two-column pages read left-then-right; paragraphs and tables that continue across a page break are stitched back together. |
| **Headings** | PDF bookmarks are used as ground truth when they match the page text. Otherwise: font size vs. body size, bold, `1.2.3` numbering, `Article / Section / Schedule` keywords, ALL CAPS. A single dominant top-level heading on page 1-2 becomes the document title. |
| **Structure** | A heading stack turns blocks into sections; each section carries its breadcrumb (`Title > 2. Loan Terms > 2.1 Events of Default`). |
| **Chunk** | Greedy packing of whole blocks, per section (see below). |
| **Export** | Text + metadata, with a breadcrumb-prefixed `contextualized_text` field intended for embedding. |

## Chunking strategy and why

**Size: target 400 tokens, hard max 512, min 100.**
- 512 is the input limit of many popular embedding models (BGE, E5, MiniLM family). The limit is applied to the *embedded* text, so the breadcrumb header's tokens are subtracted from every budget. Large-context embedders accept more, but retrieval quality generally favours focused chunks.
- 400 tokens (~300 words, 2-4 paragraphs) is large enough to hold a full legal clause or an MD&A paragraph group, and small enough that one chunk is usually about one thing.
- Anything <= max is never split, so a 480-token paragraph stays whole instead of being cut to hit 400. The target is a packing goal, not a cutting rule.
- Chunks under 100 tokens are merged with the following section when it is a child or a deeper sibling (never two unrelated top-level chapters).

**Overlap: 50 tokens(~12%), whole sentences only, within a section only.**
- Overlap exists to rescue facts that straddle a boundary ("...subject to the following. [boundary] The Borrower shall..."). 10-15% is the usual sweet spot; more mostly duplicates storage and retrieval hits.
- It is made of complete trailing sentences (or, for legal enumerations with no sentence breaks, the last `;`-delimited clause), so no sentence is ever cut in half.
- None across section boundaries (the breadcrumb already supplies context, and overlapping unrelated sections adds noise) and none for tables.
- `overlap_prefix_chars` says how many leading characters of `text` repeat the previous chunk, so you can strip them for display or dedupe.

**Sentences, clauses, paragraphs**
- Blocks are the atoms. A chunk boundary is only placed *between* paragraphs/lists/tables unless a single paragraph exceeds the max.
- Oversize paragraphs split on **sentence** boundaries (abbreviation-aware: `Rs. 5.2 crore`, `approx.`, `Mr. J. Sharma`, `U.S. GAAP`, `Inc.`, `Cl. 4.2`, but `Schedule A.` ends a sentence). A sentence that is itself too long (typical of loan agreements) splits at `;` and `(a)/(i)` enumeration markers, then commas, and only as a last resort on words. Every such case is flagged (`long_paragraph_split`, `long_sentence_split`).

**Lists**: a lead-in sentence ending in `:` stays with its items, and the list is one atom. If a list must be split, it splits between items and the lead-in is repeated on each part.

**Tables**: always their own chunks (never mixed with prose), rendered as Markdown with the caption (`Table 2: ...`). If too large, split between rows with the header row repeated, and `table.part / n_parts` recorded. The raw `header` and `rows` are kept in the `table` field so you can also index row-level or load into pandas.

## Chunk schema (one JSON object per chunk)
```
{
    "chunk_id": "...",
    "doc_id": "...", 
    "chunk_index": "...", 
    "prev_chunk_id": "...", 
    "next_chunk_id": "...", 
    "source_file": "...", 
    "doc_title": "...",
    "section_path" : "list", 
    "section_title": "...", 
    "section_level": "...", 
    "sections_covered" : "when tiny sections were merged",
    "page_start": "...", 
    "page_end" (null for DOCX/TXT): "...", 
    "content_type" : "text|list|table|mixed", 
    "element_types": "...",
    "token_count": "...", 
    "char_count": "...", 
    "token_counter": "...", 
    "overlap_prefix_chars": "...", 
    "flags": "...", 
    "table": "table chunks only",
    "text": "...", 
    "contextualized_text": "...", 
    "content_hash": "..."
}
```