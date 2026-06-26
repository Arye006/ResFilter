# Smart Candidate Hiring Assistant — Backend

Semantic resume-to-JD matching using free local sentence embeddings
(no API keys, no cost). Built with FastAPI.

## How it works (one paragraph)
We turn the job description and every resume into numeric vectors
("embeddings") using a small transformer model (`all-MiniLM-L6-v2`)
that understands meaning, not just words. We then compute **cosine
similarity** between the JD vector and each resume vector — a score
from -1 to 1 showing how aligned their meanings are — and sort
candidates from highest to lowest score.

## Setup (in VS Code)

1. Open this folder in VS Code.
2. Create a virtual environment (keeps this project's packages separate
   from everything else on your machine):
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # on Mac/Linux
   venv\Scripts\activate           # on Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```
   First run will download the embedding model (~90MB) — needs internet
   once. After that it's cached locally and works offline.

5. Open **http://127.0.0.1:8000/docs** in your browser.
   This is FastAPI's auto-generated interactive docs — you can test
   both endpoints right there, no frontend needed yet.

## Endpoints

### `GET /`
Health check. Confirms the server is alive.

### `POST /rank-text`
Send JD + resumes as plain text (JSON body):
```json
{
  "job_description": "Looking for a backend dev skilled in Python and FastAPI...",
  "resumes": {
    "Alice": "Built REST APIs in Django and Flask for 3 years...",
    "Bob": "Frontend engineer specializing in React..."
  }
}
```

### `POST /rank-files`
Send JD as a form field + resumes as uploaded PDF/DOCX/TXT files
(multipart/form-data — this is what your file-upload frontend will call).

Form fields:
- `job_description` (text)
- `resumes` (one or more files)

## Project structure
```
app/
├── main.py              # FastAPI app + routes (the entrypoint)
├── embedding_engine.py   # loads the model, computes embeddings + cosine similarity
├── text_extractor.py     # pulls text out of PDF / DOCX / TXT
├── ranker.py             # combines embedding + similarity into a sorted ranking
└── schemas.py            # request/response data shapes (pydantic models)
```


