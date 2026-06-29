"""
quiz_engine.py  —  StudyMate "Quiz me" + adaptive weak-spot targeting

Jobs:
  1. generate_quiz()   -> build a quiz from the student's notes. If they have
                          recorded weak spots, bias the quiz toward those topics.
  2. grade_answer()    -> grade one answer (MCQ by match, short by LLM judge).
  3. record_weak_spot()-> when an answer is wrong, remember the topic so future
                          quizzes can focus there.  (the USP)
  4. get_weak_spots()  -> list the student's weak topics for the "weak spots" view.

Uses the local Ollama LLM. Weak spots live in a `weak_spots` table in Supabase.
"""

import json
import ollama

from database import get_connection

CHAT_MODEL = "llama3.1"


def _get_some_notes(limit_chars=4000):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT content FROM chunks ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    text = "\n\n".join(r[0] for r in rows)
    return text[:limit_chars]


def _get_weak_topics(user_email, limit=5):
    """Return the topics this user misses most, to focus the next quiz."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT topic FROM weak_spots WHERE user_email = %s ORDER BY times_missed DESC, last_missed DESC LIMIT %s",
        (user_email, limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [r[0] for r in rows]


def generate_quiz(user_email, num_questions=4):
    """
    Build a quiz from the notes. If the student has weak spots, tell the LLM to
    focus questions on those topics - this is the adaptive part (the USP):
    the quiz targets exactly what the student keeps getting wrong.
    """
    notes = _get_some_notes()
    if not notes.strip():
        return {"error": "No notes found. Add some notes first, then come back to quiz yourself."}

    weak_topics = _get_weak_topics(user_email)
    focus_line = ""
    if weak_topics:
        focus_line = (
            "The student has struggled with these topics before, so make MORE of "
            "the questions focus on them: " + ", ".join(weak_topics) + ".\n"
        )

    prompt = f"""You are a tutor making a short quiz to test a student on THEIR OWN notes below.

Make exactly {num_questions} questions, mixing multiple-choice and short-answer.
Base every question ONLY on the notes. Keep them clear and not trick questions.
{focus_line}
For each question, also include a short "topic" label (2-4 words) naming what it tests.

Return ONLY valid JSON, no preamble, in exactly this shape:
{{
  "questions": [
    {{"type": "mcq", "topic": "...", "question": "...", "options": ["...","...","...","..."], "answer": "exact text of the correct option"}},
    {{"type": "short", "topic": "...", "question": "...", "answer": "a concise model answer"}}
  ]
}}

NOTES:
{notes}
"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    raw = response["message"]["content"]

    try:
        data = json.loads(raw)
        questions = data.get("questions", [])
        clean = []
        for q in questions:
            topic = q.get("topic", "general")
            if q.get("type") == "mcq" and q.get("options") and q.get("answer"):
                clean.append({"type": "mcq", "topic": topic, "question": q["question"], "options": q["options"], "answer": q["answer"]})
            elif q.get("type") == "short" and q.get("answer"):
                clean.append({"type": "short", "topic": topic, "question": q["question"], "answer": q["answer"]})
        if not clean:
            return {"error": "Could not build a quiz from these notes. Try adding a bit more detail."}
        return {"questions": clean, "focused_on": weak_topics}
    except json.JSONDecodeError:
        return {"error": "The quiz generator returned an unexpected format. Try again."}


def grade_answer(question_type, question, correct_answer, student_answer):
    student_answer = (student_answer or "").strip()
    if not student_answer:
        return {"correct": False, "feedback": "No answer given."}

    if question_type == "mcq":
        is_right = student_answer.strip().lower() == correct_answer.strip().lower()
        return {"correct": is_right, "feedback": "Correct!" if is_right else f"Not quite - the answer is: {correct_answer}"}

    prompt = f"""A student answered a short-answer question. Decide if their answer is essentially correct.

Question: {question}
Model answer: {correct_answer}
Student's answer: {student_answer}

Reply with ONLY valid JSON: {{"correct": true or false, "feedback": "one short sentence of feedback"}}"""

    response = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], format="json")
    try:
        data = json.loads(response["message"]["content"])
        return {"correct": bool(data.get("correct", False)), "feedback": data.get("feedback", "")}
    except json.JSONDecodeError:
        return {"correct": None, "feedback": f"Model answer: {correct_answer}"}


def record_weak_spot(user_email, topic, question):
    """
    Remember that the student missed this topic. If the topic is already in
    their weak spots, bump the count and timestamp; otherwise add it.
    This is what makes the next quiz adaptive.
    """
    if not topic:
        topic = "general"
    conn = get_connection()
    cur = conn.cursor()
    # is this topic already tracked for this user?
    cur.execute(
        "SELECT id, times_missed FROM weak_spots WHERE user_email = %s AND topic = %s",
        (user_email, topic),
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE weak_spots SET times_missed = times_missed + 1, last_missed = now() WHERE id = %s",
            (row[0],),
        )
    else:
        cur.execute(
            "INSERT INTO weak_spots (user_email, topic, question) VALUES (%s, %s, %s)",
            (user_email, topic, question),
        )
    conn.commit()
    cur.close()
    conn.close()


def get_weak_spots(user_email):
    """Return the student's weak topics, most-missed first, for the view."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT topic, times_missed FROM weak_spots WHERE user_email = %s ORDER BY times_missed DESC, last_missed DESC",
        (user_email,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"topic": r[0], "times_missed": r[1]} for r in rows]