import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from src.evaluation.models import GoldenEvaluationCase, EvaluationSuite, QueryType


class SplitManager:
    """Manages stratified train/test splitting and validation for golden evaluation suites."""

    @staticmethod
    def split_dataset(
        cases: List[GoldenEvaluationCase],
        test_size: float = 0.25,
        random_seed: int = 42
    ) -> Tuple[List[GoldenEvaluationCase], List[GoldenEvaluationCase]]:
        random.seed(random_seed)
        by_type: Dict[str, List[GoldenEvaluationCase]] = {}
        for c in cases:
            by_type.setdefault(c.query_type.value, []).append(c)

        active_cases: List[GoldenEvaluationCase] = []
        hidden_cases: List[GoldenEvaluationCase] = []

        for qtype, qcases in by_type.items():
            shuffled = list(qcases)
            random.shuffle(shuffled)
            split_point = int(len(shuffled) * (1.0 - test_size))
            for idx, c in enumerate(shuffled):
                if idx < split_point:
                    c.split = "active"
                    active_cases.append(c)
                else:
                    c.split = "hidden"
                    hidden_cases.append(c)

        return active_cases, hidden_cases

    @staticmethod
    def save_suites(
        all_cases: List[GoldenEvaluationCase],
        active_cases: List[GoldenEvaluationCase],
        hidden_cases: List[GoldenEvaluationCase],
        output_dir: str = "data/evaluation"
    ):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "golden_full.json", "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in all_cases], f, indent=2)

        with open(out_path / "active_suite.json", "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in active_cases], f, indent=2)

        with open(out_path / "hidden_suite.json", "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in hidden_cases], f, indent=2)
