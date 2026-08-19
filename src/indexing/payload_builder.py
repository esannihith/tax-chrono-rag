from typing import List, Dict, Any
from pydantic import BaseModel
from src.models import Chunk, NodeType
from src.enrichment.normalizer import TaxEntityNormalizer
from src.enrichment.table_linearizer import TableLinearizer


class IndexPayload(BaseModel):
    chunk_id: str
    corpus_year: int
    rule_id: str
    statutory_path: str
    node_type: NodeType
    dense_text: str       # Text string passed to dense embedder (with passage prefix)
    sparse_text: str      # Text string tokenized for BM25
    display_content: str  # Original / Markdown formatted text for LLM prompt
    metadata: Dict[str, Any]


class PayloadBuilder:
    """Prepares dual-channel payloads (dense semantic + sparse lexical) from raw AST chunks."""

    @classmethod
    def build_payload(cls, chunk: Chunk) -> IndexPayload:
        raw_content = chunk.content
        breadcrumb = chunk.breadcrumb
        is_table = chunk.metadata.is_table

        # 1. Normalize citations and acronyms
        norm_content = TaxEntityNormalizer.normalize_chunk_text(raw_content)

        # 2. Linearize table rows if table chunk
        if is_table:
            linear_text = TableLinearizer.linearize_markdown_table(norm_content, breadcrumb)
            dense_text = f"{breadcrumb}\n{linear_text}"
        else:
            dense_text = f"{breadcrumb}\n{norm_content}"

        # 3. Build sparse lexical representation
        sparse_text = f"{breadcrumb} {norm_content} {' '.join(chunk.metadata.sections_referenced)} {' '.join(chunk.metadata.forms_referenced)}"

        return IndexPayload(
            chunk_id=chunk.chunk_id,
            corpus_year=chunk.metadata.corpus_year,
            rule_id=chunk.metadata.rule_id,
            statutory_path=chunk.metadata.statutory_path,
            node_type=chunk.metadata.node_type,
            dense_text=dense_text,
            sparse_text=sparse_text,
            display_content=chunk.content,
            metadata=chunk.metadata.model_dump()
        )

    @classmethod
    def build_all(cls, chunks: List[Chunk]) -> List[IndexPayload]:
        return [cls.build_payload(c) for c in chunks]
