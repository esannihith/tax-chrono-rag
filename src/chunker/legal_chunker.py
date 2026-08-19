import re
from typing import List, Optional
from src.models import RuleDocument, StatutoryNode, Chunk, ChunkMetadata, NodeType
from src.chunker.breadcrumbs import BreadcrumbBuilder


class LegalChunker:
    """Structure-Aware Parent-Child Chunker:
    Combines sub-rules with their child clauses, provisos, explanations, and embedded tables
    into coherent, self-contained statutory units."""

    SECTION_RE = re.compile(r"\b(?:section|sub-section)\s+\d+[A-Za-z]*(?:\([0-9A-Za-z]+\))*", re.IGNORECASE)
    FORM_RE = re.compile(r"\b(?:Form\s+(?:No\.\s*)?(?:ITR-[0-9A-Z]+|SUGAM|SAHAJ|[0-9A-Z]+))\b", re.IGNORECASE)

    @classmethod
    def extract_references(cls, text: str) -> tuple[List[str], List[str]]:
        sections = list(set(cls.SECTION_RE.findall(text)))
        forms = list(set(cls.FORM_RE.findall(text)))
        return sorted(sections), sorted(forms)

    @classmethod
    def extract_node_full_text(cls, node: StatutoryNode) -> str:
        parts = []
        if node.content and node.content.strip():
            parts.append(node.content.strip())
        for child in node.children:
            child_text = cls.extract_node_full_text(child)
            if child_text:
                parts.append(child_text)
        return "\n".join(parts)

    @classmethod
    def chunk_document(cls, doc: RuleDocument) -> List[Chunk]:
        chunks = []
        rule_prefix = f"[Income Tax Rules {doc.corpus_year} > Rule {doc.rule_id}: {doc.title}]"

        doc_eff_date = None
        for fn in doc.footnotes:
            if fn.effective_date:
                doc_eff_date = fn.effective_date
                break

        total_len = sum(len(cls.extract_node_full_text(rn)) for rn in doc.root_nodes)

        # Rule-Level Chunking (for compact rules <= 4500 chars)
        if total_len <= 4500 or len(doc.root_nodes) <= 1:
            full_content = "\n\n".join(cls.extract_node_full_text(rn) for rn in doc.root_nodes if cls.extract_node_full_text(rn))
            if not full_content.strip():
                full_content = doc.clean_text.strip()

            chunk_id = f"chunk_{doc.corpus_year}_rule_{doc.rule_id}_full"
            secs, forms = cls.extract_references(full_content)

            meta = ChunkMetadata(
                chunk_id=chunk_id,
                parent_id=f"rule_{doc.corpus_year}_{doc.rule_id}",
                corpus_year=doc.corpus_year,
                rule_id=doc.rule_id,
                rule_title=doc.title,
                statutory_path=f"Rule {doc.rule_id}",
                node_type=NodeType.RULE,
                source_file=doc.source_file,
                sections_referenced=secs,
                forms_referenced=forms,
                effective_date=doc_eff_date,
                is_table=any(rn.node_type == NodeType.TABLE for rn in doc.root_nodes)
            )

            chunks.append(Chunk(
                chunk_id=chunk_id,
                breadcrumb=rule_prefix,
                text=f"{rule_prefix}\n{full_content}",
                content=full_content,
                metadata=meta
            ))
            return chunks

        # Sub-Rule / Major Node Level Chunking (for long rules like Rule 2BB, Rule 3, Rule 280)
        for rn in doc.root_nodes:
            rn_text = cls.extract_node_full_text(rn)
            if not rn_text.strip():
                continue

            node_path = f"Rule {doc.rule_id} > {rn.label}" if rn.label else f"Rule {doc.rule_id}"
            node_breadcrumb = f"[Income Tax Rules {doc.corpus_year} > Rule {doc.rule_id}: {doc.title} > {rn.label}]" if rn.label else rule_prefix
            chunk_id = f"chunk_{doc.corpus_year}_rule_{doc.rule_id}_{rn.node_id}"

            secs, forms = cls.extract_references(rn_text)

            node_eff_date = doc_eff_date
            for fn_id in rn.footnotes_referenced:
                matching_fn = next((f for f in doc.footnotes if f.id == fn_id), None)
                if matching_fn and matching_fn.effective_date:
                    node_eff_date = matching_fn.effective_date
                    break

            meta = ChunkMetadata(
                chunk_id=chunk_id,
                parent_id=f"rule_{doc.corpus_year}_{doc.rule_id}",
                corpus_year=doc.corpus_year,
                rule_id=doc.rule_id,
                rule_title=doc.title,
                statutory_path=node_path,
                node_type=rn.node_type,
                source_file=doc.source_file,
                sections_referenced=secs,
                forms_referenced=forms,
                effective_date=node_eff_date,
                is_table=(rn.node_type == NodeType.TABLE)
            )

            chunks.append(Chunk(
                chunk_id=chunk_id,
                breadcrumb=node_breadcrumb,
                text=f"{node_breadcrumb}\n{rn_text}",
                content=rn_text,
                metadata=meta
            ))

        return chunks
