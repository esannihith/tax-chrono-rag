import re
from typing import List, Optional


class TableLinearizer:
    """Converts tabular legal rows into rich declarative sentences for dense embeddings."""

    @classmethod
    def linearize_markdown_table(cls, markdown_table: str, breadcrumb: str) -> str:
        lines = [l.strip() for l in markdown_table.split("\n") if l.strip()]
        if len(lines) < 3:
            return markdown_table

        # Parse header
        headers = [c.strip() for c in lines[0].split("|")[1:-1]]
        headers = [h for h in headers if h]

        row_sentences = []
        for line in lines[2:]:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if not any(cells):
                continue
            
            pairs = []
            for h, c in zip(headers, cells):
                if c and c != "-":
                    pairs.append(f"{h}: {c}")
            if pairs:
                row_sentences.append("Row entry where " + "; ".join(pairs) + ".")

        if row_sentences:
            return "\n".join(row_sentences)
        return markdown_table
