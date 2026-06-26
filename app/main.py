"""
main.py
--------
This is the entrypoint of the backend. Run it with:
    uvicorn app.main:app --reload

It exposes two ways to rank candidates against a job description:
1. POST /rank-text   -> JD and resumes sent as plain text (JSON)
2. POST /rank-files  -> JD as text, resumes as uploaded PDF/DOCX files

Once running, visit http://127.0.0.1:8000/docs for interactive,
auto-generated API documentation where you can test everything
in the browser without writing a single line of frontend code.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import TextRankRequest, RankResponse
from app.text_extractor import extract_text
from app.ranker import rank_candidates

app = FastAPI(
    title="Smart Candidate Hiring Assistant API",
    description="Semantic resume-to-JD matching using sentence embeddings.",
    version="1.0.0",
)

# CORS lets your friend's frontend (running on a different port/domain)
# call this API from the browser. For a hackathon, allowing all origins
# is fine — just don't ship this wide-open setting to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    """Simple endpoint to confirm the server is alive."""
    return {"status": "ok", "message": "Smart Hiring Assistant backend is running."}


@app.post("/rank-text", response_model=RankResponse)
def rank_text(payload: TextRankRequest):
    """
    Rank candidates when JD and resumes are provided as plain text.

    Example JSON body:
    {
        "job_description": "Looking for a backend dev skilled in Python and FastAPI...",
        "resumes": {
            "Alice": "Built REST APIs in Django and Flask for 3 years...",
            "Bob": "Frontend engineer specializing in React and Tailwind..."
        }
    }
    """
    if not payload.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty.")

    results = rank_candidates(payload.job_description, payload.resumes)

    return RankResponse(
        job_description_preview=payload.job_description[:150],
        total_candidates=len(results),
        results=results,
    )


@app.post("/rank-files", response_model=RankResponse)
async def rank_files(
    job_description: str = Form(..., description="Job description as plain text"),
    resumes: list[UploadFile] = File(..., description="Resume files (PDF or DOCX)"),
):
    """
    Rank candidates when resumes are uploaded as PDF/DOCX files.
    job_description is still plain text (a form field, not a file),
    since JDs are almost always typed/pasted, not uploaded.
    """
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="job_description cannot be empty.")

    resume_texts: dict[str, str] = {}

    for upload in resumes:
        file_bytes = await upload.read()
        try:
            text = extract_text(upload.filename, file_bytes)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract any text from '{upload.filename}'.",
            )

        # Use the filename (without extension) as the candidate's display name
        candidate_name = upload.filename.rsplit(".", 1)[0]
        resume_texts[candidate_name] = text

    results = rank_candidates(job_description, resume_texts)

    return RankResponse(
        job_description_preview=job_description[:150],
        total_candidates=len(results),
        results=results,
    )
