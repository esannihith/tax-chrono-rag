from typing import List


class BreadcrumbBuilder:
    """Generates hierarchical statutory breadcrumbs for context injection."""

    @staticmethod
    def build(year: int, rule_id: str, rule_title: str, path_segments: List[str]) -> str:
        base = f"Income Tax Rules {year} > Rule {rule_id}: {rule_title}"
        if path_segments:
            return f"[{base} > {' > '.join(path_segments)}]"
        return f"[{base}]"
