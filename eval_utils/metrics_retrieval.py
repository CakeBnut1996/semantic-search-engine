import re
from typing import List, Set, Dict, Any
# Import the data model from your existing retriever file
from retrieval_utils.retriever import RetrievalResult


class MetricCalculator:
    """
    Calculates statistical metrics for retrieval (MRR, Hit@K, Coverage).
    Does NOT require an LLM.
    """

    @staticmethod
    def _get_tokens(text: str) -> Set[str]:
        return set(re.findall(r'\w+', text.lower()))

    @staticmethod
    def calculate_coverage(gold_quote: str, chunk_text: str) -> float:
        """Calculates percentage of gold tokens present in retrieved chunk."""
        gold_tokens = MetricCalculator._get_tokens(gold_quote)
        chunk_tokens = MetricCalculator._get_tokens(chunk_text)

        if not gold_tokens:
            return 0.0

        intersection = gold_tokens.intersection(chunk_tokens)
        return len(intersection) / len(gold_tokens)

    @staticmethod
    def evaluate_positives(
            retrieved_items: List[RetrievalResult],
            gold_ds_id: str,
            gold_quote: str
    ) -> Dict[str, Any]:
        """
        Evaluates retrieval when we expect a specific document (Positive Test).
        """
        # 1. Dataset Level Metrics
        seen_datasets = []
        for r in retrieved_items:
            if r.dataset_id not in seen_datasets:
                seen_datasets.append(r.dataset_id)

        try:
            best_ds_rank = seen_datasets.index(gold_ds_id) + 1
        except ValueError:
            best_ds_rank = None

        ds_stats = {
            "ds_hit_1": 1 if best_ds_rank == 1 else 0,
            "ds_hit_3": 1 if best_ds_rank and best_ds_rank <= 3 else 0,
            "ds_mrr": (1 / best_ds_rank) if best_ds_rank else 0
        }

        # 2. Chunk Level Metrics (based on text overlap/coverage)
        best_ch_rank = None
        max_coverage = 0.0

        for r in retrieved_items:
            coverage = MetricCalculator.calculate_coverage(gold_quote, r.chunk_text)
            max_coverage = max(max_coverage, coverage)

            # We consider it a "hit" if coverage > 70%
            if coverage >= 0.7 and best_ch_rank is None:
                best_ch_rank = r.rank  # Assuming rank is available or inferred from index+1

        # If rank isn't explicitly in the object, assume list order:
        if best_ch_rank is None:
            # Logic to find rank if not set in object, optional fallback
            pass

        ch_stats = {
            "chunk_hit_1": 1 if best_ch_rank == 1 else 0,
            "chunk_hit_3": 1 if best_ch_rank and best_ch_rank <= 3 else 0,
            "chunk_mrr": (1 / best_ch_rank) if best_ch_rank else 0,
            "max_coverage_found": round(max_coverage, 4),
            "top_retrieval_distance": retrieved_items[0].score if retrieved_items else None
        }

        return {**ds_stats, **ch_stats}

    @staticmethod
    def evaluate_negatives(retrieved_items: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Evaluates retrieval when we expect NOTHING relevant (Hard Negative Test).
        Mainly checks confidence/distance scores.
        """
        return {
            "top_retrieval_distance": retrieved_items[0].score if retrieved_items else None
        }