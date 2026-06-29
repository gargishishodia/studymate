"""
rag_engine.py  —  StudyMate core RAG loop (Supabase version, corrected)

Five steps:
  1. chunk_text       split a long text into smaller overlapping pieces
  2. embed            turn a piece of text into a vector (768 numbers)
  3. add_chunks       INSERT each piece + vector into Postgres        # STORE
  4. search           SQL similarity search with pgvector's <=>       # STORE
  5. answer_question  feed retrieved pieces to the LLM, get an answer

Prereqs: Ollama running with `nomic-embed-text` and `llama3.1` pulled, and a
working SUPABASE_DB_URL in your .env (test it with: python database.py).

pip install ollama numpy psycopg2-binary pgvector python-dotenv
"""

import ollama
from database import get_connection   # our Supabase connection helper

EMBED_MODEL = "nomic-embed-text"   # outputs a 768-number vector
CHAT_MODEL = "llama3.1"            # writes the final answer


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
    """Map text to a 768-dim vector. The SAME model must embed both the
    documents and the later questions, or the vectors aren't comparable."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


# ----------------------------------------------------------------------
# 3. ADD CHUNKS - writes to Supabase.                         # STORE
# ----------------------------------------------------------------------
def add_chunks(chunks):
    """
    For each chunk: embed it, then INSERT (content + embedding) into the
    `chunks` table in Supabase.

    Note conn.commit() - Postgres holds changes in a transaction until you
    commit, which is when they actually get saved.
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
    Embed the question, then let Postgres find the closest chunks.

    The `<=>` operator computes COSINE DISTANCE between the stored embeddings
    and our question vector. ORDER BY that distance (smallest first) LIMIT k
    gives the k most similar chunks.

    We pass the vector as a string and cast it with `::vector` so Postgres
    treats it as a real vector type (without the cast it sees a plain numeric
    array and the <=> operator doesn't apply).
    """
    question_vector = embed(question)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT content FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
        (str(question_vector), k),
    )
    rows = cur.fetchall()       # list of (content,) tuples
    cur.close()
    conn.close()
    return [row[0] for row in rows]


# ----------------------------------------------------------------------
# 5. ANSWER - retrieve, stuff into prompt, generate.
# ----------------------------------------------------------------------
def answer_question(question, k=3):
    """Retrieve relevant chunks, put them in the prompt, and ask the LLM to
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

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response["message"]["content"]
    return answer, context_chunks


# ----------------------------------------------------------------------
# DEMO - runs only when you execute this file directly.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_notes = """
    Photosynthesis is the process by which green plants convert light energy
    into chemical energy. It takes place mainly in the leaves, inside
    structures called chloroplasts, which contain the green pigment chlorophyll.
    Chlorophyll absorbs light, mostly in the blue and red parts of the spectrum,
    and reflects green light, which is why plants look green.

    The overall reaction takes carbon dioxide and water and, using light energy,
    produces glucose and oxygen. The glucose stores energy for the plant, while
    the oxygen is released into the air as a by-product.

    Photosynthesis has two main stages. The light-dependent reactions happen in
    the thylakoid membranes and capture energy from sunlight. The light-
    independent reactions, also called the Calvin cycle, happen in the stroma
    and use that captured energy to build glucose from carbon dioxide.
    """

    print("Step 1: chunking the notes...")
    chunks = chunk_text(sample_notes)
    print(f"  -> got {len(chunks)} chunks\n")

    print("Step 2 & 3: embedding and storing in Supabase...")
    add_chunks(chunks)
    print()

    questions = [
        "Why do plants look green?",
        "Where does the Calvin cycle happen?",
        "What does photosynthesis produce?",
    ]

    for q in questions:
        print("=" * 60)
        print("Q:", q)
        answer, sources = answer_question(q)
        print("\nA:", answer)
        print("\n(Answer came from these chunks:)")
        for i, s in enumerate(sources, 1):
            preview = s[:80].replace("\n", " ")
            print(f"  [{i}] {preview}...")
        print()