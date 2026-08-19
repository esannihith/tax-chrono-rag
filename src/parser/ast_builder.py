import re
from typing import List, Dict, Any, Optional, Tuple
from src.models import StatutoryNode, NodeType, TableData, Footnote, RuleDocument
from src.parser.cleaner import RuleCleaner
from src.parser.table_extractor import TableExtractor


class ASTBuilder:
    """Builds an Indian Statutory Legal Abstract Syntax Tree (AST)."""

    SUB_RULE_RE = re.compile(r"^\((\d+[A-Za-z]*)\)\s*(.*)$")
    CLAUSE_RE = re.compile(r"^\(([a-z]+)\)\s*(.*)$")
    SUB_CLAUSE_RE = re.compile(r"^\(([ivxlcdm]+)\)\s*(.*)$")
    ITEM_RE = re.compile(r"^\(([A-Z]+)\)\s*(.*)$")
    PROVISO_RE = re.compile(r"^(Provided(?:\s+further|\s+also)?\s+that\s*)(.*)$", re.IGNORECASE)
    EXPLANATION_RE = re.compile(r"^(Explanation(?:\s+\d+)?\.?\s*[—\-\:]?)\s*(.*)$", re.IGNORECASE)
    SCHEDULE_OR_PART_RE = re.compile(r"^(PART\s+[IVXLCDM]+|SCHEDULE\s+[IVXLCDM]+)", re.IGNORECASE)

    RULE_HEADING_RE = re.compile(r"^(\d+[A-Z]*)\.\s*(.*)$")

    @classmethod
    def clean_footnote_markers(cls, text: str) -> Tuple[str, List[int]]:
        fn_ids = []
        def replace_fn(match):
            fn_id = int(match.group(1))
            fn_ids.append(fn_id)
            return match.group(2)

        cleaned = re.sub(r"(\d+)\[([^\]]*)\]", replace_fn, text)
        # Also clean leading unclosed footnote marker e.g. 33[Updated return -> Updated return
        unclosed_match = re.match(r"^(\d+)\[(.*)$", cleaned)
        if unclosed_match:
            fn_ids.append(int(unclosed_match.group(1)))
            cleaned = unclosed_match.group(2)

        cleaned = re.sub(r"\[\*\*\*\]", "", cleaned)
        cleaned = re.sub(r"\]", "", cleaned)
        return cleaned.strip(), fn_ids

    @classmethod
    def extract_rule_title_and_body(cls, lines: List[str], rule_id: str) -> Tuple[str, List[str]]:
        title = ""
        body_lines = []
        found_rule_start = False

        for i, line in enumerate(lines):
            if cls.SCHEDULE_OR_PART_RE.match(line):
                continue

            rule_match = re.search(rf"\b{re.escape(rule_id)}\.", line)
            if rule_match:
                found_rule_start = True
                before_rule = line[:rule_match.start()].strip()
                after_rule = line[rule_match.end():].strip()

                if before_rule:
                    clean_t, _ = cls.clean_footnote_markers(before_rule)
                    title = clean_t.strip(" [].")
                elif i > 0:
                    clean_t, _ = cls.clean_footnote_markers(lines[i-1])
                    title = clean_t.strip(" [].")

                if after_rule:
                    body_lines.append(after_rule)
                continue

            if found_rule_start:
                body_lines.append(line)
            else:
                if not title and len(line) < 120:
                    clean_t, _ = cls.clean_footnote_markers(line)
                    title = clean_t.strip(" [].")

        if not found_rule_start:
            body_lines = lines

        return title, body_lines

    @classmethod
    def classify_line(cls, line: str) -> Tuple[NodeType, str, str]:
        m = cls.SUB_RULE_RE.match(line)
        if m:
            return NodeType.SUB_RULE, f"({m.group(1)})", m.group(2).strip()

        m = cls.CLAUSE_RE.match(line)
        if m:
            return NodeType.CLAUSE, f"({m.group(1)})", m.group(2).strip()

        m = cls.SUB_CLAUSE_RE.match(line)
        if m:
            return NodeType.SUB_CLAUSE, f"({m.group(1)})", m.group(2).strip()

        m = cls.ITEM_RE.match(line)
        if m:
            return NodeType.ITEM, f"({m.group(1)})", m.group(2).strip()

        m = cls.PROVISO_RE.match(line)
        if m:
            return NodeType.PROVISO, m.group(1).strip(), m.group(2).strip()

        m = cls.EXPLANATION_RE.match(line)
        if m:
            return NodeType.EXPLANATION, m.group(1).strip(), m.group(2).strip()

        return NodeType.OTHER, "", line.strip()

    @classmethod
    def build_ast(cls, parsed_doc_dict: Dict[str, Any]) -> RuleDocument:
        rule_id = parsed_doc_dict["rule_id"]
        year = parsed_doc_dict["corpus_year"]
        source_file = parsed_doc_dict["source_file"]
        pages = parsed_doc_dict["pages"]

        all_clean_lines = []
        all_footnotes: List[Footnote] = []

        for p in pages:
            raw_lines = p["raw_text"].split("\n") if p["raw_text"] else []
            clean_lines, fns = RuleCleaner.clean_page_lines(raw_lines)
            all_clean_lines.extend(clean_lines)
            all_footnotes.extend(fns)

        unique_footnotes = {f.id: f for f in all_footnotes}
        sorted_footnotes = sorted(unique_footnotes.values(), key=lambda x: x.id)

        title, body_lines = cls.extract_rule_title_and_body(all_clean_lines, rule_id)
        if not title:
            title = f"Rule {rule_id}"

        tables = TableExtractor.extract_tables_from_pages(pages)

        root_nodes: List[StatutoryNode] = []
        current_subrule: Optional[StatutoryNode] = None
        current_clause: Optional[StatutoryNode] = None
        current_subclause: Optional[StatutoryNode] = None
        current_item: Optional[StatutoryNode] = None

        node_counter = 0

        def get_node_id(prefix: str) -> str:
            nonlocal node_counter
            node_counter += 1
            return f"rule_{rule_id}_{prefix}_{node_counter}"

        for line in body_lines:
            cleaned_text, fn_refs = cls.clean_footnote_markers(line)
            if not cleaned_text:
                continue

            node_type, label, rest = cls.classify_line(cleaned_text)

            if node_type == NodeType.SUB_RULE:
                current_subrule = StatutoryNode(
                    node_id=get_node_id(f"subrule_{label.strip('()')}"),
                    node_type=NodeType.SUB_RULE,
                    label=label,
                    content=rest,
                    footnotes_referenced=fn_refs,
                    depth=1
                )
                root_nodes.append(current_subrule)
                current_clause = None
                current_subclause = None
                current_item = None

            elif node_type == NodeType.CLAUSE:
                current_clause = StatutoryNode(
                    node_id=get_node_id(f"clause_{label.strip('()')}"),
                    node_type=NodeType.CLAUSE,
                    label=label,
                    content=rest,
                    footnotes_referenced=fn_refs,
                    depth=2 if current_subrule else 1
                )
                if current_subrule:
                    current_subrule.children.append(current_clause)
                else:
                    root_nodes.append(current_clause)
                current_subclause = None
                current_item = None

            elif node_type == NodeType.SUB_CLAUSE:
                current_subclause = StatutoryNode(
                    node_id=get_node_id(f"subclause_{label.strip('()')}"),
                    node_type=NodeType.SUB_CLAUSE,
                    label=label,
                    content=rest,
                    footnotes_referenced=fn_refs,
                    depth=3 if current_subrule else 2
                )
                if current_clause:
                    current_clause.children.append(current_subclause)
                elif current_subrule:
                    current_subrule.children.append(current_subclause)
                else:
                    root_nodes.append(current_subclause)
                current_item = None

            elif node_type == NodeType.ITEM:
                current_item = StatutoryNode(
                    node_id=get_node_id(f"item_{label.strip('()')}"),
                    node_type=NodeType.ITEM,
                    label=label,
                    content=rest,
                    footnotes_referenced=fn_refs,
                    depth=4 if current_subclause else 3
                )
                if current_subclause:
                    current_subclause.children.append(current_item)
                elif current_clause:
                    current_clause.children.append(current_item)
                elif current_subrule:
                    current_subrule.children.append(current_item)
                else:
                    root_nodes.append(current_item)

            elif node_type in (NodeType.PROVISO, NodeType.EXPLANATION):
                spec_node = StatutoryNode(
                    node_id=get_node_id(node_type.value.lower()),
                    node_type=node_type,
                    label=label,
                    content=rest,
                    footnotes_referenced=fn_refs,
                    depth=1
                )
                if current_subrule:
                    current_subrule.children.append(spec_node)
                else:
                    root_nodes.append(spec_node)

            else:
                target_node = current_item or current_subclause or current_clause or current_subrule
                if target_node:
                    target_node.content += " " + cleaned_text
                    target_node.footnotes_referenced.extend(fn_refs)
                else:
                    root_nodes.append(StatutoryNode(
                        node_id=get_node_id("body"),
                        node_type=NodeType.RULE,
                        label=f"Rule {rule_id}",
                        content=cleaned_text,
                        footnotes_referenced=fn_refs,
                        depth=0
                    ))

        if tables:
            for t_idx, table in enumerate(tables):
                table_node = StatutoryNode(
                    node_id=get_node_id(f"table_{t_idx+1}"),
                    node_type=NodeType.TABLE,
                    label=f"Table {t_idx+1}",
                    content=table.markdown,
                    table=table,
                    depth=1
                )
                root_nodes.append(table_node)

        return RuleDocument(
            rule_id=rule_id,
            corpus_year=year,
            title=title,
            source_file=source_file,
            page_count=len(pages),
            root_nodes=root_nodes,
            footnotes=sorted_footnotes,
            extracted_tables=tables,
            clean_text="\n".join(all_clean_lines)
        )
