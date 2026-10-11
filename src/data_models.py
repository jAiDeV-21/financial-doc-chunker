from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Block:
    kind: str  # heading | paragraph | footnote | list | table | subheading
    text: str = ""
    page: Optional[int] = None
    page_end: Optional[int] = None
    level: int = 0  # headings only (1 = top)
    intro: str = ""  # lists: lead-in sentence ("... the following:")
    items: list = field(default_factory=list)  # lists
    header: list = field(default_factory=list)  # tables
    rows: list = field(default_factory=list)  # tables
    caption: str = ""  # tables


@dataclass
class ParsedDoc:
    title: str
    blocks: list
    n_pages: Optional[int]
    info: dict


@dataclass
class Section:
    path: list
    level: int
    blocks: list
    page: Optional[int] = None
    covered: list = field(
        default_factory=list
    )  # titles of all sections merged into this one


@dataclass
class Unit:
    """Atomic piece the packer never splits."""

    text: str
    tokens: int
    kind: str  # prose | list | table
    page: Optional[int] = None
    page_end: Optional[int] = None
    join_prev: str = (
        "\n\n"  # separator before this unit (" " = same paragraph continues)
    )
    flags: set = field(default_factory=set)
    table: Optional[dict] = None
    etypes: set = field(default_factory=set)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    source_file: str
    doc_title: str
    section_path: list
    section_title: str
    section_level: int
    sections_covered: list
    page_start: Optional[int]
    page_end: Optional[int]
    content_type: str  # text | list | table | mixed
    element_types: list
    token_count: int
    char_count: int
    overlap_prefix_chars: int
    flags: list
    prev_chunk_id: Optional[str]
    next_chunk_id: Optional[str]
    table: Optional[dict]
    text: str  # clean chunk content (what you show / feed to the LLM)
    contextualized_text: str  # breadcrumb + text (what you should EMBED)
    content_hash: str
    token_counter: str = ""
