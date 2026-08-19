import re
from typing import List, Set
from src.models import RuleDocument, StatutoryNode, Chunk, ChunkMetadata, NodeType
from src.chunker.breadcrumbs import BreadcrumbBuilder


class LegalChunker:
    """Produces context-enriched, semantically atomic chunks from RuleDocument ASTs."""

    SECTION_RE = re.compile(r"\b(?:section|sub-section)\s+\d+[A-Za-z]*(?:\([0-9A-Za-z]+\))*", re.IGNORECASE)
    FORM_RE = re.compile(r"\b(?:Form\s+(?:No\.\s*)?(?:ITR-[0-9A-Z]+|SUGAM|SAHAJ|[0-9A-Z]+))\b", re.IGNORECASE)

    @classmethod
    def extract_references(cls, text: str) -> tuple[List[str], List[str]]:
        sections = list(set(cls.SECTION_RE.findall(text)))
        forms = list(set(cls.FORM_RE.findall(text)))
        return sorted(sections), sorted(forms)

    @classmethod
    def chunk_node(
        cls,
        doc: RuleDocument,
        node: StatutoryNode,
        parent_paths: List[str]
    ) -> List[Chunk]:
        chunks = []
        current_path = list(parent_paths)
        if node.label:
            current_path.append(node.label)

        if node.content and node.content.strip():
            breadcrumb = BreadcrumbBuilder.build(
                year=doc.corpus_year,
                rule_id=doc.rule_id,
                rule_title=doc.title,
                path_segments=current_path
            )
            full_text = f"{breadcrumb}\n{node.content}"
            secs, forms = cls.extract_references(node.content)

            chunk_id = f"chunk_{doc.corpus_year}_rule_{doc.rule_id}_{node.node_id}"
            statutory_path = " > ".join(current_path) if current_path else f"Rule {doc.rule_id}"

            eff_date = None
            for fn_id in node.footnotes_referenced:
                matching_fn = next((f for f in doc.footnotes if f.id == fn_id), None)
                if matching_fn and matching_fn.effective_date:
                    eff_date = matching_fn.effective_date
                    break

            chunks.append(Chunk(
                chunk_id=chunk_id,
                breadcrumb=breadcrumb,
                text=full_text,
                content=node.content,
                metadata=ChunkMetadata(
                    chunk_id=chunk_id,
                    corpus_year=doc.corpus_year,
                    rule_id=doc.rule_id,
                    rule_title=doc.title,
                    statutory_path=statutory_path,
                    node_type=node.node_type,
                    source_file=doc.source_file,
                    sections_referenced=secs,
                    forms_referenced=forms,
                    effective_date=eff_date,
                    is_table=(node.node_type == NodeType.TABLE)
                )
            ))

        for child in node.children:
            chunks.extend(cls.chunk_node(doc, child, current_path))

        return chunks

    @classmethod
    def chunk_document(cls, doc: RuleDocument) -> List[Chunk]:
        all_chunks = []
        for root_node in doc.root_nodes:
            all_chunks.extend(cls.chunk_node(doc, root_node, []))
        return all_chunks
