"""
main.py  —  StudyMate web API (FastAPI)

Endpoints:
  POST /auth/signup    create an account            (from auth.py)
  POST /auth/login     log in, get a JWT            (from auth.py)
  POST /documents      upload text (PROTECTED)
  POST /query          ask a question (PROTECTED)
  POST /quiz/generate  make a quiz from my notes (PROTECTED)
  POST /quiz/grade     grade one answer (PROTECTED)

Run:   uvicorn main:app --reload
Open:  http://localhost:8000/docs
"""

import os

from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import rag_engine
import quiz_engine
from auth import router as auth_router, get_current_user

app = FastAPI(title="StudyMate API")

# --- CORS (deploy-ready) ---
# Which website origins are allowed to call this API. We read them from the
# ALLOWED_ORIGINS env var (comma-separated) so the SAME code works in dev and
# in production without edits:
#   - Local dev: defaults below allow file:// pages and localhost.
#   - Production: set ALLOWED_ORIGINS in your host's env to your real frontend
#     URL, e.g.  ALLOWED_ORIGINS=https://studymate.vercel.app
#
# Note: when the frontend is opened as a local file, the browser Origin is
# "null". We include it for convenience during local development only.
_default_origins = "null,http://localhost,http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500"
_origins = os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
ALLOWED_ORIGINS = [o.strip() for o in _origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(auth_router)


# --- Request shapes with input limits ---
# max_length caps how much text a request can carry, so nobody can overload the
# server with a giant payload. These are generous for real notes but bounded.
class DocumentIn(BaseModel):
    text: str = Field(min_length=1, max_length=50000)


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=3, ge=1, le=10)


class QuizGenIn(BaseModel):
    num_questions: int = Field(default=4, ge=1, le=10)


class GradeIn(BaseModel):
    type: str = Field(max_length=10)
    topic: str = Field(default="general", max_length=100)
    question: str = Field(max_length=2000)
    correct_answer: str = Field(max_length=2000)
    student_answer: str = Field(max_length=2000)


@app.get("/")
def home():
    return {"status": "StudyMate API is running"}


# --- write path (protected) ---
def process_document(text: str):
    chunks = rag_engine.chunk_text(text)
    rag_engine.add_chunks(chunks)
    print(f"[background] stored {len(chunks)} chunks")


@app.post("/documents")
def upload_document(
    doc: DocumentIn,
    background_tasks: BackgroundTasks,
    user: str = Depends(get_current_user),
):
    background_tasks.add_task(process_document, doc.text)
    return {"status": "accepted", "message": f"Document is being processed for {user}."}


# --- read path (protected) ---
@app.post("/query")
def ask_question(q: QueryIn, user: str = Depends(get_current_user)):
    answer, sources = rag_engine.answer_question(q.question, k=q.k)
    return {"answer": answer, "sources": sources}


# --- quiz: generate (protected) ---
@app.post("/quiz/generate")
def quiz_generate(data: QuizGenIn, user: str = Depends(get_current_user)):
    # `user` is the email from the token - used to focus the quiz on THIS
    # student's weak spots (the adaptive part).
    return quiz_engine.generate_quiz(user_email=user, num_questions=data.num_questions)


# --- quiz: grade one answer (protected) ---
@app.post("/quiz/grade")
def quiz_grade(data: GradeIn, user: str = Depends(get_current_user)):
    result = quiz_engine.grade_answer(
        question_type=data.type,
        question=data.question,
        correct_answer=data.correct_answer,
        student_answer=data.student_answer,
    )
    # If the student got it wrong, remember the topic so future quizzes can
    # focus on it. This is what makes the quiz adaptive (the USP).
    if result.get("correct") is False:
        quiz_engine.record_weak_spot(user_email=user, topic=data.topic, question=data.question)
    return result


# --- weak spots: what the student keeps getting wrong (protected) ---
@app.get("/quiz/weak-spots")
def quiz_weak_spots(user: str = Depends(get_current_user)):
    return {"weak_spots": quiz_engine.get_weak_spots(user)}
