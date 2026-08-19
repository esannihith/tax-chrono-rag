import re
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel


class NormalizedQuery(BaseModel):
    raw_query: str
    dense_query: str
    sparse_tokens: List[str]
    detected_fy: Optional[str] = None
    detected_ay: Optional[str] = None
    target_rules: List[str] = []


class TaxEntityNormalizer:
    """Canonicalizes tax citations, expands domain acronyms, and resolves FY/AY mappings."""

    ACRONYM_MAP = {
        r"\bHRA\b": "House Rent Allowance (HRA)",
        r"\bLTC\b": "Leave Travel Concession (LTC)",
        r"\bLTA\b": "Leave Travel Allowance (LTA)",
        r"\bITR-U\b|\bITRU\b": "Updated Return of Income (Form ITR-U)",
        r"\bEVC\b": "Electronic Verification Code (EVC)",
        r"\bDSC\b": "Digital Signature Certificate (DSC)",
        r"\bVRS\b": "Voluntary Retirement Scheme (VRS)",
        r"\bNPS\b": "National Pension System (NPS)",
        r"\bEPF\b": "Employees Provident Fund (EPF)",
        r"\bTDS\b": "Tax Deducted at Source (TDS)",
        r"\bTCS\b": "Tax Collected at Source (TCS)",
        r"\bNew\s+Regime\b|\bNew\s+Tax\s+Regime\b": "New Tax Regime (section 115BAC / section 202(4))",
        r"\bOld\s+Regime\b|\bOld\s+Tax\s+Regime\b": "Old Tax Regime (regular provisions)"
    }

    CITATION_PATTERNS = [
        (re.compile(r"\bu/s\s*(\d+[A-Za-z]*(?:\([0-9A-Za-z]+\))*)(?:\s+of\s+the\s+Act)?", re.IGNORECASE), r"section \1"),
        (re.compile(r"\bsec\.?\s*(\d+[A-Za-z]*(?:\([0-9A-Za-z]+\))*)(?:\s+of\s+the\s+Act)?", re.IGNORECASE), r"section \1"),
        (re.compile(r"\brule\s*(\d+[A-Za-z]*)", re.IGNORECASE), r"Rule \1"),
        (re.compile(r"\bform\s*(\d+[A-Za-z]*)", re.IGNORECASE), r"Form No. \1"),
        (re.compile(r"\bsch(?:edule)?\.?\s*([ivxlcdm\d]+)", re.IGNORECASE), r"Schedule \1")
    ]

    FY_PATTERN = re.compile(r"\b(?:FY|Financial\s+Year)\s*(\d{4})[-–/](\d{2,4})\b", re.IGNORECASE)
    AY_PATTERN = re.compile(r"\b(?:AY|Assessment\s+Year)\s*(\d{4})[-–/](\d{2,4})\b", re.IGNORECASE)

    @classmethod
    def canonicalize_citations(cls, text: str) -> str:
        res = text
        for pat, repl in cls.CITATION_PATTERNS:
            res = pat.sub(repl, res)
        return res

    @classmethod
    def expand_acronyms(cls, text: str) -> str:
        res = text
        for pat_str, expanded in cls.ACRONYM_MAP.items():
            res = re.sub(pat_str, expanded, res, flags=re.IGNORECASE)
        return res

    @classmethod
    def resolve_fy_ay(cls, text: str) -> Tuple[str, Optional[str], Optional[str]]:
        res = text
        detected_fy = None
        detected_ay = None

        # Check FY e.g. FY 2020-21 -> AY 2021-22
        def replace_fy(m):
            nonlocal detected_fy, detected_ay
            y1 = int(m.group(1))
            detected_fy = f"FY {y1}-{y1+1}"
            detected_ay = f"AY {y1+1}-{y1+2}"
            return f"Financial Year {y1}-{y1+1} (Assessment Year {y1+1}-{y1+2})"

        # Check AY e.g. AY 2021-22 -> FY 2020-21
        def replace_ay(m):
            nonlocal detected_ay, detected_fy
            y1 = int(m.group(1))
            detected_ay = f"AY {y1}-{y1+1}"
            detected_fy = f"FY {y1-1}-{y1}"
            return f"Assessment Year {y1}-{y1+1} (Financial Year {y1-1}-{y1})"

        res = cls.FY_PATTERN.sub(replace_fy, res)
        res = cls.AY_PATTERN.sub(replace_ay, res)
        return res, detected_fy, detected_ay

    @classmethod
    def normalize_chunk_text(cls, text: str) -> str:
        text = cls.canonicalize_citations(text)
        text = cls.expand_acronyms(text)
        return text

    @classmethod
    def normalize_query(cls, query: str) -> NormalizedQuery:
        norm_text = cls.canonicalize_citations(query)
        norm_text = cls.expand_acronyms(norm_text)
        norm_text, fy, ay = cls.resolve_fy_ay(norm_text)

        # Extract targeted rule mentions
        rules = re.findall(r"\bRule\s+(\d+[A-Za-z]*)", norm_text, re.IGNORECASE)

        # Build sparse tokens (raw tokens + expanded tokens without stopwords)
        tokens = [t.strip(" ,.:;()[]?\"'") for t in re.split(r"\s+", norm_text) if len(t.strip(" ,.:;()[]?\"'")) > 1]

        return NormalizedQuery(
            raw_query=query,
            dense_query=norm_text,
            sparse_tokens=list(set(tokens)),
            detected_fy=fy,
            detected_ay=ay,
            target_rules=sorted(list(set(rules)))
        )
