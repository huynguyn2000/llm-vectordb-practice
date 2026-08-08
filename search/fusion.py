"""Reciprocal Rank Fusion — merge several ranked ID lists into one ranking.

RRF is scale-free: it uses each item's rank position, not its raw score, so
rankings from different engines (cosine similarity, ts_rank) combine without
normalization. Reference: Cormack et al., TREC 2009.
"""


def reciprocal_rank_fusion(
    ranked_id_lists: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ranked ID lists. Each ID scores Σ 1/(k + rank) over the lists it
    appears in (rank is 0-based). Returns (id, score) sorted by score
    descending, breaking ties by ascending id (deterministic)."""
    scores: dict[int, float] = {}
    for id_list in ranked_id_lists:
        for rank, doc_id in enumerate(id_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
