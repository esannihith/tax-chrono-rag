import os
import re
from typing import Dict, Any, List, Tuple
import pdfplumber


class PDFLoader:
    """Loads PDF files, extracts metadata, character coordinates, and structure trees."""

    @staticmethod
    def parse_filename(filepath: str) -> Tuple[str, int]:
        """Extracts rule_id and year from filename (e.g. Rule-12AC_1962.pdf -> ('12AC', 1962))."""
        basename = os.path.basename(filepath)
        match = re.match(r"^Rule-([A-Za-z0-9]+)_(\d{4})\.pdf$", basename)
        if match:
            return match.group(1), int(match.group(2))
        return "UNKNOWN", 0

    @classmethod
    def load_pdf(cls, filepath: str) -> Dict[str, Any]:
        rule_id, year = cls.parse_filename(filepath)
        pages_data = []

        with pdfplumber.open(filepath) as pdf:
            pdf_metadata = pdf.metadata or {}
            for page_idx, page in enumerate(pdf.pages):
                st = getattr(page, "structure_tree", None)
                if callable(st):
                    st = st()

                # Build mcid -> chars map
                mcid_chars = {}
                for char in page.chars:
                    m = char.get("mcid")
                    if m is not None:
                        mcid_chars.setdefault(m, []).append(char)

                mcid_text = {
                    m: "".join(c["text"] for c in chars)
                    for m, chars in mcid_chars.items()
                }

                raw_text = page.extract_text(layout=False) or ""
                layout_text = page.extract_text(layout=True) or ""
                tables = page.extract_tables() or []

                pages_data.append({
                    "page_number": page_idx + 1,
                    "width": page.width,
                    "height": page.height,
                    "structure_tree": st,
                    "mcid_text": mcid_text,
                    "chars": page.chars,
                    "raw_text": raw_text,
                    "layout_text": layout_text,
                    "extracted_tables": tables
                })

        return {
            "source_file": filepath,
            "rule_id": rule_id,
            "corpus_year": year,
            "page_count": len(pages_data),
            "metadata": pdf_metadata,
            "pages": pages_data
        }
