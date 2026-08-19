import os
from typing import Dict, Any, List
from src.models import RuleDocument
from src.parser.pdf_loader import PDFLoader
from src.parser.ast_builder import ASTBuilder


class ParserPipeline:
    """Orchestrates PDF ingestion, noise removal, table extraction, and AST generation."""

    @classmethod
    def parse_file(cls, filepath: str) -> RuleDocument:
        raw_doc_dict = PDFLoader.load_pdf(filepath)
        rule_doc = ASTBuilder.build_ast(raw_doc_dict)
        return rule_doc

    @classmethod
    def parse_directory(cls, dirpath: str) -> List[RuleDocument]:
        docs = []
        for fname in sorted(os.listdir(dirpath)):
            if fname.endswith(".pdf"):
                fpath = os.path.join(dirpath, fname)
                doc = cls.parse_file(fpath)
                docs.append(doc)
        return docs
