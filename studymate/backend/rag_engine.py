"""
rag_engine.py  —  StudyMate core RAG loop (Gemini version, deploy-ready)

Five steps:
  1. chunk_text       split a long text into smaller overlapping pieces
  2. embed            turn a piece of text into a vector (768 numbers)
  3. add_chunks       INSERT each piece + vector into Postgres        # STORE
  4. search           SQL similarity search with pgvector's <=>       # STORE
  5. answer_question  feed retrieved pieces to the LLM, get an answer

Uses Google Gemini (free tier) for BOTH embeddings and chat, so the app can be
deployed to a normal host (no local Ollama needed).

Setup:
  pip install google-genai psycopg2-binary pgvector python-dotenv
  Set GEMINI_API_KEY and SUPABASE_DB_URL in your .env
"""

import os
from google import genai
from google.genai import types

from database import get_connection   # our Supabase connection helper

# One Gemini client for the whole module. It automatically reads the
# GEMINI_API_KEY environment variable (loaded from .env by database.py).
client = genai.Client()

EMBED_MODEL = "gemini-embedding-001"   # we request 768 dims to match our table
EMBED_DIM = 768
CHAT_MODEL = "gemini-2.5-flash"        # free-tier chat model, writes answers


# ----------------------------------------------------------------------
# 1. CHUNKING - split text into overlapping pieces.
# ----------------------------------------------------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping pieces so each embedding is specific,
    and ideas split across a boundary still appear whole in a neighbour."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


# ----------------------------------------------------------------------
# 2. EMBEDDING - same model for documents AND questions.
# ----------------------------------------------------------------------
def embed(text):
    """Map text to a 768-dim vector with Gemini. The SAME model must embed
    both the documents and the later questions, or the vectors aren't
    comparable. We force 768 dims so it matches our `vector(768)` column."""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    # The SDK returns an object with .embeddings (a list); each has .values
    return response.embeddings[0].values


# ----------------------------------------------------------------------
# 3. ADD CHUNKS - writes to Supabase.                         # STORE
# ----------------------------------------------------------------------
def add_chunks(chunks):
    """
    For each chunk: embed it, then INSERT (content + embedding) into the
    `chunks` table in Supabase. We pass the vector as a string and cast with
    `::vector` so Postgres stores it as a real vector.
    """
    conn = get_connection()
    cur = conn.cursor()
    for chunk in chunks:
        vector = embed(chunk)
        cur.execute(
            "INSERT INTO chunks (content, embedding) VALUES (%s, %s::vector)",
            (chunk, str(vector)),
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Stored {len(chunks)} chunks in Supabase.")


# ----------------------------------------------------------------------
# 4. SEARCH - one SQL query using pgvector.                   # STORE
# ----------------------------------------------------------------------
def search(question, k=3):
    """
    Embed the question, then let Postgres find the closest chunks using the
    `<=>` cosine-distance operator. ORDER BY distance LIMIT k returns the k
    most similar chunks. The `::vector` cast makes Postgres treat our string
    as a real vector so the operator applies.
    """
    question_vector = embed(question)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT content FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
        (str(question_vector), k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in rows]


# ----------------------------------------------------------------------
# 5. ANSWER - retrieve, stuff into prompt, generate.
# ----------------------------------------------------------------------
def answer_question(question, k=3):
    """Retrieve relevant chunks, put them in the prompt, and ask Gemini to
    answer using ONLY that context. Grounding stops it making things up and
    lets us show which chunks the answer came from (citations)."""
    context_chunks = search(question, k=k)
    context = "\n\n".join(context_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the notes."

Context:
{context}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )
    answer = response.text
    return answer, context_chunks