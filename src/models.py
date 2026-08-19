from __future__ import annotations
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    RULE = "RULE"
    SUB_RULE = "SUB_RULE"
    CLAUSE = "CLAUSE"
    SUB_CLAUSE = "SUB_CLAUSE"
    ITEM = "ITEM"
    SUB_ITEM = "SUB_ITEM"
    PROVISO = "PROVISO"
    EXPLANATION = "EXPLANATION"
    TABLE = "TABLE"
    SCHEDULE = "SCHEDULE"
    DEFINITION = "DEFINITION"
    OTHER = "OTHER"


class Footnote(BaseModel):
    id: int
    text: str
    amendment_type: Optional[str] = None
    rules_referenced: Optional[str] = None
    effective_date: Optional[str] = None


class TableData(BaseModel):
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    caption: Optional[str] = None
    markdown: str = ""


class StatutoryNode(BaseModel):
    node_id: str
    node_type: NodeType
    label: str = ""
    title: Optional[str] = None
    content: str = ""
    table: Optional[TableData] = None
    children: List[StatutoryNode] = Field(default_factory=list)
    footnotes_referenced: List[int] = Field(default_factory=list)
    depth: int = 0


class RuleDocument(BaseModel):
    rule_id: str
    corpus_year: int
    title: str
    source_file: str
    page_count: int
    root_nodes: List[StatutoryNode] = Field(default_factory=list)
    footnotes: List[Footnote] = Field(default_factory=list)
    extracted_tables: List[TableData] = Field(default_factory=list)
    clean_text: str = ""


class ChunkMetadata(BaseModel):
    chunk_id: str
    parent_id: Optional[str] = None
    corpus_year: int
    rule_id: str
    rule_title: str
    statutory_path: str
    node_type: NodeType
    source_file: str
    sections_referenced: List[str] = Field(default_factory=list)
    forms_referenced: List[str] = Field(default_factory=list)
    effective_date: Optional[str] = None
    is_table: bool = False


class Chunk(BaseModel):
    chunk_id: str
    breadcrumb: str
    text: str
    content: str
    metadata: ChunkMetadata
