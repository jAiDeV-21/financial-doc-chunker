from data_models import Block, Section


def build_sections(blocks: list[Block]) -> list[Section]:
    stack: list[tuple[int, str]] = []
    cur = Section([], 0, [])
    sections = [cur]
    for b in blocks:
        if b.kind == "heading":
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.text))
            cur = Section([t for _, t in stack], b.level, [], b.page)
            sections.append(cur)
        else:
            cur.blocks.append(b)
    return sections
