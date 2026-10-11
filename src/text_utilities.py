import re
import unicodedata

_LIGATURES = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb00": "ff",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\u200b\u200c\u200d\ufeff\u00ad]")


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    for k, v in _LIGATURES.items():
        s = s.replace(k, v)
    s = _CTRL.sub("", s).replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", s).strip()


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines into one paragraph, repairing end-of-line hyphenation."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if (
            out.endswith("-")
            and len(out) > 1
            and out[-2].isalpha()
            and ln[:1].islower()
        ):
            out = out[:-1] + ln
        else:
            out = f"{out} {ln}" if out else ln
    return clean_text(out)


_LIST_MARK = re.compile(
    r"^\s*(?:[•●▪■◦○‣∙·*–—-]\s+|\((?:[a-zA-Z]{1,4}|\d{1,3})\)\s+|(?:[a-zA-Z]|\d{1,3})[.)]\s+)"
)
_HEAD_NUM = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2}){0,4}[.)]?|[A-Z][.)]|[IVXLC]{1,6}[.)])\s+\S"
)
_HEAD_KW = re.compile(
    r"^(?:article|section|part|chapter|schedule|annex(?:ure)?|appendix|exhibit|recitals?|"
    r"definitions|notes? to)\b",
    re.IGNORECASE,
)
_TOC_LEADER = re.compile(r"(?:\.\s?){5,}\s*\d*\s*$")
_PAGE_NO = re.compile(
    r"^(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?$|^[-–—]\s*\d{1,4}\s*[-–—]$",
    re.IGNORECASE,
)


def numbering_depth(t: str) -> int:
    m = re.match(r"^(\d{1,2}(?:\.\d{1,2})*)", t)
    return m.group(1).count(".") + 1 if m else 0


def looks_like_heading(
    t: str, bold: bool = False, ratio: float = 1.0, plain: bool = False
) -> bool:
    words = len(t.split())
    if (
        not t
        or words > 25
        or len(t) > 200
        or t.endswith((",", ";"))
        or _TOC_LEADER.search(t)
    ):
        return False
    caps = t.isupper() and sum(c.isalpha() for c in t) >= 3 and words <= 14
    numbered = bool(_HEAD_NUM.match(t)) and words <= 18
    keyword = bool(_HEAD_KW.match(t)) and words <= 14
    ends_sentence = t.endswith((".", "!", "?"))
    if ratio >= 1.12:
        return True
    if bold and words <= 18 and (numbered or not ends_sentence):
        return True
    if (numbered or keyword) and (bold or caps):
        return True
    if caps and words <= 12 and not t.endswith("."):
        return True
    return bool(plain and (numbered or keyword) and words <= 12 and not ends_sentence)


# ---- sentence splitting (abbreviation aware: "Rs. 5.2 crore", "Mr. Rao", "approx. 5%") ----
_ABBR = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "no",
    "nos",
    "vs",
    "etc",
    "inc",
    "ltd",
    "co",
    "corp",
    "pvt",
    "plc",
    "llc",
    "llp",
    "lp",
    "approx",
    "est",
    "fig",
    "figs",
    "vol",
    "art",
    "sec",
    "secs",
    "cl",
    "para",
    "paras",
    "sch",
    "e.g",
    "i.e",
    "cf",
    "viz",
    "ca",
    "dept",
    "govt",
    "rs",
    "incl",
    "excl",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "u.s",
    "u.k",
    "p.a",
    "a.m",
    "p.m",
    "ref",
    "tel",
    "ph",
    "mt",
    "ave",
    "w.e.f",
    "a/c",
}
_BOUNDARY = re.compile(r"""([.!?]["”’')\]]*)(\s+)(?=["“‘(\[]?[A-Z0-9₹$€£])""")


def split_sentences(text: str) -> list[str]:
    sents, start = [], 0
    for m in _BOUNDARY.finditer(text):
        cand = text[start : m.end(1)]
        toks = cand.split()
        if m.group(1).startswith(".") and toks:
            raw = toks[-1].rstrip(".!?\"”’')]").lstrip('("“‘[')
            core = raw.lower()
            prev_has_dot = len(toks) > 1 and toks[-2].endswith(".")
            if (
                core in _ABBR
                or re.fullmatch(r"[a-z]", raw)  # "e." / "p." fragments
                or re.fullmatch(r"(?:[a-z]\.)+[a-z]", core)  # u.s / e.g / i.e
                or (
                    re.fullmatch(r"[A-Z]", raw) and prev_has_dot
                )  # "Mr. J." initials, not "Schedule A."
                or re.fullmatch(r"\(?[A-Za-z0-9]{1,4}[.)]", cand.strip())
            ):
                continue
        sents.append(cand.strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sents.append(tail)
    return sents


# Split points for oversize sentences, strongest first (legal enumerations before commas)
_SPLIT_LEVELS = [
    re.compile(r"(?<=;)\s+|\s+(?=\((?:[a-z]{1,4}|\d{1,2})\)\s)|(?<=:)\s+(?=\()"),
    re.compile(r"(?<=,)\s+|\s+(?=[—–]\s)|(?<=:)\s+"),
]
