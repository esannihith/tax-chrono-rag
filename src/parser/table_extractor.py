from typing import List, Dict, Any, Optional
from src.models import TableData
import re


class TableExtractor:
    """Extracts and formats tables from PDF structure trees and layout coordinates."""

    @staticmethod
    def format_markdown_table(headers: List[str], rows: List[List[str]], caption: Optional[str] = None) -> str:
        clean_headers = [re.sub(r"\s+", " ", h or "").strip() for h in headers]
        if not clean_headers and rows:
            clean_headers = [f"Col {i+1}" for i in range(len(rows[0]))]

        col_count = len(clean_headers)
        md_lines = []
        if caption:
            md_lines.append(f"**Table: {caption}**")
            md_lines.append("")

        header_row = "| " + " | ".join(clean_headers) + " |"
        sep_row = "| " + " | ".join(["---"] * col_count) + " |"
        md_lines.append(header_row)
        md_lines.append(sep_row)

        for row in rows:
            clean_row = []
            for cell in row:
                cell_text = re.sub(r"\s+", " ", str(cell) if cell is not None else "").strip()
                cell_text = cell_text.replace("|", "\\|")
                clean_row.append(cell_text)
            if len(clean_row) < col_count:
                clean_row.extend([""] * (col_count - len(clean_row)))
            else:
                clean_row = clean_row[:col_count]
            md_lines.append("| " + " | ".join(clean_row) + " |")

        return "\n".join(md_lines)

    @classmethod
    def extract_tables_from_pages(cls, pages: List[Dict[str, Any]]) -> List[TableData]:
        extracted = []
        for page in pages:
            for raw_table in page.get("extracted_tables", []):
                if not raw_table or len(raw_table) == 0:
                    continue

                valid_rows = [
                    [str(c).strip() if c is not None else "" for c in r]
                    for r in raw_table
                    if any(c is not None and str(c).strip() for c in r)
                ]
                if not valid_rows:
                    continue

                max_cols = max(len(r) for r in valid_rows)
                normalized_rows = [r + [""] * (max_cols - len(r)) for r in valid_rows]

                headers = normalized_rows[0]
                rows = normalized_rows[1:] if len(normalized_rows) > 1 else []

                if len(normalized_rows) >= 2 and all(re.match(r"^\(?\d+\)?$", c) for c in normalized_rows[1] if c):
                    headers = [f"{h} {idx}".strip() for h, idx in zip(normalized_rows[0], normalized_rows[1])]
                    rows = normalized_rows[2:]

                md = cls.format_markdown_table(headers, rows)
                extracted.append(TableData(
                    headers=headers,
                    rows=rows,
                    markdown=md
                ))

        return extracted
