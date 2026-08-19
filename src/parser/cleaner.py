import re
from typing import List, Tuple, Dict
from src.models import Footnote


class RuleCleaner:
    """Strips portal headers, footers, and separates footnote amendment blocks."""

    HEADER_PATTERNS = [
        re.compile(r"^Income Tax Department$", re.IGNORECASE),
        re.compile(r"^Ministry of Finance, Government of India$", re.IGNORECASE),
        re.compile(r"^Government of India$", re.IGNORECASE)
    ]

    FOOTER_PATTERN = re.compile(
        r"^Downloaded/Printed on .* from www\.incometaxindia\.gov\.in\s+Page \d+ of \d+$",
        re.IGNORECASE
    )

    FOOTNOTE_PATTERN = re.compile(
        r"^(\d+)\.\s+(Inserted|Substituted|Omitted|Amended|Prior to its substitution|Prior to their substitution|Prior to its omission)\s+(.*)$",
        re.DOTALL
    )

    WEF_PATTERN = re.compile(r"w\.e\.f\.\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", re.IGNORECASE)
    WREF_PATTERN = re.compile(r"w\.r\.e\.f\.\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", re.IGNORECASE)

    @classmethod
    def is_header(cls, line: str) -> bool:
        trimmed = line.strip()
        return any(p.match(trimmed) for p in cls.HEADER_PATTERNS)

    @classmethod
    def is_footer(cls, line: str) -> bool:
        return bool(cls.FOOTER_PATTERN.match(line.strip()))

    @classmethod
    def parse_footnote(cls, line: str) -> Tuple[bool, Footnote | None]:
        match = cls.FOOTNOTE_PATTERN.match(line.strip())
        if not match:
            return False, None
        fn_id = int(match.group(1))
        amendment_type = match.group(2)
        rest = match.group(3).strip()

        eff_date = None
        wef = cls.WEF_PATTERN.search(rest)
        wref = cls.WREF_PATTERN.search(rest)
        if wef:
            eff_date = wef.group(1)
        elif wref:
            eff_date = wref.group(1)

        return True, Footnote(
            id=fn_id,
            text=line.strip(),
            amendment_type=amendment_type,
            rules_referenced=rest,
            effective_date=eff_date
        )

    @classmethod
    def clean_page_lines(cls, lines: List[str]) -> Tuple[List[str], List[Footnote]]:
        clean_lines = []
        footnotes = []

        in_footnote_section = False
        current_footnote: Footnote | None = None

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if cls.is_header(line) or cls.is_footer(line):
                continue

            # Check if this line starts a footnote
            is_fn, fn_obj = cls.parse_footnote(line)
            if is_fn and fn_obj:
                in_footnote_section = True
                if current_footnote:
                    footnotes.append(current_footnote)
                current_footnote = fn_obj
                continue

            if in_footnote_section:
                # Continuation of multi-line footnote
                if current_footnote:
                    current_footnote.text += " " + line
                continue

            clean_lines.append(line)

        if current_footnote:
            footnotes.append(current_footnote)

        return clean_lines, footnotes
