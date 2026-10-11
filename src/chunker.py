import hashlib
from collections.abc import Callable

from config import ChunkConfig
from data_models import Block, Chunk, ParsedDoc, Section, Unit
from structure import build_sections
from text_utilities import _SPLIT_LEVELS, split_sentences


class Chunker:
    def __init__(self, cfg: ChunkConfig, count: Callable[[str], int]):
        self.cfg, self.count = cfg, count

    # ---------------------------------------------------------------- helpers
    def _limits(self, doc_title: str, path: list) -> tuple[int, int, int]:
        """(target, max, header_tokens) with the breadcrumb header budgeted in."""
        hdr = self.count(" > ".join([doc_title] + path)) + 2
        t = max(self.cfg.target_tokens - hdr, self.cfg.target_tokens // 2)
        m = max(self.cfg.max_tokens - hdr, self.cfg.max_tokens // 2)
        return t, m, hdr

    def _block_tokens(self, blocks: list[Block]) -> int:
        n = 0
        for b in blocks:
            if b.kind == "table":
                n += self.count("\n".join(" | ".join(r) for r in [b.header] + b.rows))
            elif b.kind == "list":
                n += self.count("\n".join([b.intro] + b.items))
            else:
                n += self.count(b.text)
        return n

    def _split_to_limit(self, text: str, limit: int, level: int = 0) -> list[str]:
        """Split an oversize span: ';' / '(a)' markers -> commas -> words. Greedy pack."""
        if self.count(text) <= limit:
            return [text]
        if level >= len(_SPLIT_LEVELS):
            pieces, cur = [], ""
            for w in text.split():
                cand = f"{cur} {w}".strip()
                if cur and self.count(cand) > limit:
                    pieces.append(cur)
                    cur = w
                else:
                    cur = cand
            return pieces + ([cur] if cur else [])
        parts = [p for p in _SPLIT_LEVELS[level].split(text) if p and p.strip()]
        if len(parts) <= 1:
            return self._split_to_limit(text, limit, level + 1)
        pieces, cur = [], ""
        for p in parts:
            if self.count(p) > limit:
                if cur:
                    pieces.append(cur)
                    cur = ""
                pieces += self._split_to_limit(p, limit, level + 1)
                continue
            cand = f"{cur} {p}".strip() if cur else p
            if cur and self.count(cand) > limit:
                pieces.append(cur)
                cur = p
            else:
                cur = cand
        return pieces + ([cur] if cur else [])

    # ---------------------------------------------------------------- blocks -> units
    def _table_units(self, b: Block, t_eff: int, m_eff: int, prefix: str) -> list[Unit]:
        esc = lambda c: c.replace("|", "\\|")  # noqa: E731
        row_md = lambda r: "| " + " | ".join(esc(c) for c in r) + " |"  # noqa: E731
        head = [row_md(b.header), "| " + " | ".join("---" for _ in b.header) + " |"]
        lead = [x for x in (prefix, b.caption) if x]

        def render(rows):
            return "\n".join(lead + head + [row_md(r) for r in rows])

        def unit(rows, part, n_parts):
            u = Unit(render(rows), 0, "table", b.page, b.page_end, etypes={"table"})
            u.tokens = self.count(u.text)
            u.table = {
                "caption": b.caption,
                "header": b.header,
                "rows": rows,
                "part": part,
                "n_parts": n_parts,
            }
            if n_parts > 1:
                u.flags.add("table_split")
            return u

        if self.count(render(b.rows)) <= m_eff:
            return [unit(b.rows, 1, 1)]
        base = self.count(render([]))
        groups, cur, tok = [], [], base
        for r in b.rows:  # split between rows, header repeated
            rt = self.count(row_md(r))
            if cur and tok + rt > t_eff:
                groups.append(cur)
                cur, tok = [], base
            cur.append(r)
            tok += rt
        if cur:
            groups.append(cur)
        return [unit(g, i + 1, len(groups)) for i, g in enumerate(groups)]

    def _list_units(self, b: Block, t_eff: int, m_eff: int, prefix: str) -> list[Unit]:
        lead = "\n".join(x for x in (prefix, b.intro) if x)
        full = "\n".join(([lead] if lead else []) + b.items)
        if self.count(full) <= m_eff:
            return [
                Unit(
                    full, self.count(full), "list", b.page, b.page_end, etypes={"list"}
                )
            ]
        items: list[str] = []
        for it in b.items:  # an item that is itself too long
            items += self._split_to_limit(it, t_eff) if self.count(it) > m_eff else [it]
        base = self.count(lead)
        groups, cur, tok = [], [], base
        for it in items:
            t = self.count(it)
            if cur and tok + t > t_eff:
                groups.append(cur)
                cur, tok = [], base
            cur.append(it)
            tok += t
        if cur:
            groups.append(cur)
        units = []
        for g in groups:  # lead-in repeated on every part
            text = "\n".join(([lead] if lead else []) + g)
            units.append(
                Unit(
                    text,
                    self.count(text),
                    "list",
                    b.page,
                    b.page_end,
                    flags={"list_split"},
                    etypes={"list"},
                )
            )
        return units

    def _to_units(self, blocks: list[Block], t_eff: int, m_eff: int) -> list[Unit]:
        units: list[Unit] = []
        prefix = ""
        for b in blocks:
            if b.kind == "subheading":  # heading of a merged tiny section
                prefix = b.text
                continue
            if b.kind == "table":
                units += self._table_units(b, t_eff, m_eff, prefix)
            elif b.kind == "list":
                units += self._list_units(b, t_eff, m_eff, prefix)
            else:
                text = f"{prefix}\n{b.text}" if prefix else b.text
                et = {"footnote"} if b.kind == "footnote" else {"paragraph"}
                n = self.count(text)
                if n <= m_eff:
                    units.append(Unit(text, n, "prose", b.page, b.page_end, etypes=et))
                else:  # long paragraph -> sentence units
                    first = True
                    for s in split_sentences(text):
                        pieces = (
                            [s]
                            if self.count(s) <= m_eff
                            else self._split_to_limit(s, t_eff)
                        )
                        for k, p in enumerate(pieces):
                            u = Unit(
                                p,
                                self.count(p),
                                "prose",
                                b.page,
                                b.page_end,
                                join_prev="\n\n" if first else " ",
                                etypes=et,
                                flags={"long_paragraph_split"},
                            )
                            if len(pieces) > 1:
                                u.flags.add("long_sentence_split")
                                if k > 0:
                                    u.join_prev = " "
                            units.append(u)
                            first = False
            prefix = ""
        return units

    # ---------------------------------------------------------------- packing
    def _tail(self, units: list[Unit], budget: int) -> str:
        """Last whole sentences of the chunk (<= budget tokens), for overlap."""
        picked: list[tuple[str, str]] = []
        used = 0
        for u in reversed(units):
            if u.kind != "prose":
                break
            sents: list[str] = []
            for s in split_sentences(u.text):  # sentences, then clauses if too big
                if self.count(s) > budget:
                    sents += [c for c in _SPLIT_LEVELS[0].split(s) if c and c.strip()]
                else:
                    sents.append(s)
            for k in range(len(sents) - 1, -1, -1):
                t = self.count(sents[k])
                if used + t > budget:
                    # a single trailing sentence slightly over budget is still worth carrying
                    if not picked and t <= 1.5 * budget:
                        picked.append((sents[k], " " if k > 0 else u.join_prev))
                    return self._join_tail(picked)
                picked.append((sents[k], " " if k > 0 else u.join_prev))
                used += t
        return self._join_tail(picked)

    @staticmethod
    def _join_tail(picked: list[tuple[str, str]]) -> str:
        seq = picked[::-1]
        if not seq:
            return ""
        out = seq[0][0]
        for s, sep in seq[1:]:
            out += sep + s
        return out

    @staticmethod
    def _assemble(units: list[Unit], overlap: str) -> str:
        s = ""
        for k, u in enumerate(units):
            s += (u.join_prev if k else "") + u.text
        return f"{overlap}{units[0].join_prev}{s}" if overlap else s

    def _pack(self, units: list[Unit], t_eff: int, m_eff: int) -> list[dict]:
        groups: list[dict] = []
        cur: list[Unit] = []
        ov, ov_tok, cur_tok = "", 0, 0

        def flush():
            nonlocal cur, ov, ov_tok, cur_tok
            if cur:
                groups.append({"units": cur, "overlap": ov, "ov_tok": ov_tok})
                ov = self._tail(cur, self.cfg.overlap_tokens)
                ov_tok = self.count(ov) if ov else 0
            cur, cur_tok = [], ov_tok

        for u in units:
            if u.kind == "table":  # tables are always standalone
                flush()
                groups.append({"units": [u], "overlap": "", "ov_tok": 0})
                ov, ov_tok, cur_tok = "", 0, 0
                continue
            if (
                cur
                and cur_tok + u.tokens > t_eff
                and not (cur_tok < self.cfg.min_tokens and cur_tok + u.tokens <= m_eff)
            ):
                flush()
            if not cur and ov and ov_tok + u.tokens > m_eff:
                ov, ov_tok, cur_tok = "", 0, 0  # never let overlap cause an overflow
            cur.append(u)
            cur_tok += u.tokens
        flush()

        # fold a tiny trailing chunk back into its predecessor
        if len(groups) >= 2:
            last, prev = groups[-1], groups[-2]
            if last["units"][0].kind != "table" and prev["units"][0].kind != "table":
                lt = sum(u.tokens for u in last["units"])
                pt = prev["ov_tok"] + sum(u.tokens for u in prev["units"])
                if lt < self.cfg.min_tokens and pt + lt <= m_eff:
                    prev["units"] += last["units"]
                    groups.pop()
                    prev["flags"] = {"merged_tail"}
        return groups

    # ---------------------------------------------------------------- sections
    def _merge_small_sections(
        self, sections: list[Section], doc_title: str
    ) -> list[Section]:
        out: list[Section] = []
        n, i = len(sections), 0
        while i < n:
            cur = sections[i]
            if not cur.blocks:  # heading-only: lives on in children's path
                i += 1
                continue
            blocks = list(cur.blocks)
            tok = self._block_tokens(blocks)
            t_eff, _, _ = self._limits(doc_title, cur.path)
            covered = [cur.path[-1]] if cur.path else []
            j = i + 1
            while (
                cur.level > 0
                and tok < self.cfg.min_tokens
                and j < n
                and sections[j].blocks
                and sections[j].level >= cur.level
                and (sections[j].level > cur.level or cur.level >= 2)
            ):
                nxt = sections[j]
                nt = self._block_tokens(nxt.blocks) + self.count(nxt.path[-1])
                if tok + nt > t_eff:
                    break
                blocks.append(Block("subheading", nxt.path[-1]))
                blocks += nxt.blocks
                tok += nt
                covered.append(nxt.path[-1])
                j += 1
            out.append(Section(cur.path, cur.level, blocks, cur.page, covered))
            i = j
        return out

    def chunk_document(
        self, parsed: ParsedDoc, doc_id: str, source_file: str, counter_name: str
    ) -> list[Chunk]:
        sections = self._merge_small_sections(
            build_sections(parsed.blocks), parsed.title
        )
        chunks: list[Chunk] = []
        for sec in sections:
            t_eff, m_eff, _ = self._limits(parsed.title, sec.path)
            groups = self._pack(self._to_units(sec.blocks, t_eff, m_eff), t_eff, m_eff)
            crumb = " > ".join([parsed.title] + sec.path)
            for g in groups:
                units: list[Unit] = g["units"]
                text = self._assemble(units, g["overlap"])
                etypes = sorted(set().union(*(u.etypes for u in units)))
                kinds = {u.kind for u in units}
                ctype = (
                    "table"
                    if kinds == {"table"}
                    else (
                        "list"
                        if kinds == {"list"}
                        else ("text" if kinds == {"prose"} else "mixed")
                    )
                )
                flags = set().union(*(u.flags for u in units)) | g.get("flags", set())
                if len(sec.covered) > 1:
                    flags.add("merged_small_sections")
                pages = [p for u in units for p in (u.page, u.page_end) if p]
                chunks.append(
                    Chunk(
                        chunk_id="",
                        doc_id=doc_id,
                        chunk_index=0,
                        source_file=source_file,
                        doc_title=parsed.title,
                        section_path=sec.path,
                        section_title=sec.path[-1] if sec.path else parsed.title,
                        section_level=sec.level,
                        sections_covered=sec.covered if len(sec.covered) > 1 else [],
                        page_start=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                        content_type=ctype,
                        element_types=etypes,
                        token_count=self.count(text),
                        char_count=len(text),
                        overlap_prefix_chars=(
                            len(g["overlap"]) + len(units[0].join_prev)
                            if g["overlap"]
                            else 0
                        ),
                        flags=sorted(flags),
                        prev_chunk_id=None,
                        next_chunk_id=None,
                        table=units[0].table if ctype == "table" else None,
                        text=text,
                        contextualized_text=f"{crumb}\n\n{text}",
                        content_hash=hashlib.sha1(text.encode("utf-8")).hexdigest()[
                            :16
                        ],
                        token_counter=counter_name,
                    )
                )
        for k, c in enumerate(chunks):
            c.chunk_index = k
            c.chunk_id = f"{doc_id}-{k:05d}"
            c.prev_chunk_id = chunks[k - 1].chunk_id if k else None
            c.next_chunk_id = chunks[k + 1].chunk_id if k + 1 < len(chunks) else None
        return chunks
