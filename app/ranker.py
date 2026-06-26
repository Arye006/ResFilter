"""
ranker.py
----------
Glue logic: given a JD and a dict of {candidate_name: resume_text},
embed everything, score it, and return a sorted (best-first) ranking.
"""

from app.embedding_engine import embed_texts, compute_similarity
from app.schemas import CandidateResult


def rank_candidates(job_description: str, resumes: dict[str, str]) -> list[CandidateResult]:
    """
    resumes: dict mapping candidate_name -> resume_text
    Returns: list of CandidateResult, sorted best-match first
    """
    if not resumes:
        return []

    candidate_names = list(resumes.keys())
    resume_texts = list(resumes.values())

    # Embed JD and all resumes in one batched call for speed
    all_texts = [job_description] + resume_texts
    all_embeddings = embed_texts(all_texts)

    jd_embedding = all_embeddings[0]
    resume_embeddings = all_embeddings[1:]

    scores = compute_similarity(jd_embedding, resume_embeddings)

    # Pair names with scores, then sort descending by score
    paired = list(zip(candidate_names, scores))
    paired.sort(key=lambda pair: pair[1], reverse=True)

    results = []
    for rank, (name, score) in enumerate(paired, start=1):
        results.append(
            CandidateResult(
                candidate_name=name,
                similarity_score=round(float(score), 4),
                rank=rank,
            )
        )
    return results
