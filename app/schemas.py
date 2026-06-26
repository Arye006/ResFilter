"""
schemas.py
-----------
Pydantic models define the "shape" of data going in and out of the API.

Why bother? Two big wins for a beginner:
1. FastAPI automatically validates incoming requests against these shapes.
   If the frontend sends bad data, you get a clear error instead of a crash.
2. FastAPI auto-generates interactive API docs (at /docs) FROM these models.
"""

from pydantic import BaseModel, Field


class CandidateResult(BaseModel):
    """One ranked candidate in the response."""
    candidate_name: str
    similarity_score: float = Field(..., description="Cosine similarity, 0 to 1")
    rank: int


class RankResponse(BaseModel):
    """Full response returned after ranking resumes against a JD."""
    job_description_preview: str
    total_candidates: int
    results: list[CandidateResult]


class TextRankRequest(BaseModel):
    """
    Used when JD and resumes are sent as plain pasted text
    (no file upload) — e.g. JSON body from the frontend.
    """
    job_description: str
    resumes: dict[str, str] = Field(
        ..., description="Mapping of candidate_name -> resume text"
    )
