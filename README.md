# Role-Aware Agent

A retrieval-augmented multi-agent system that answers questions about a company's internal documents differently depending on who's asking — an Intern gets plain language, an Engineer gets technical depth, a Manager gets a bullet-point business summary. Same facts, different framing, every answer cited back to its source.

**Live demo:** https://role-aware-agent.onrender.com
*(hosted on a free instance — the first request after a period of inactivity may take ~30–50s to wake up)*

## What makes this more than a basic RAG wrapper

- **Hybrid retrieval** — FAISS semantic search and BM25 keyword search run in parallel and get merged with Reciprocal Rank Fusion, so exact terms/numbers (which embeddings can blur) and paraphrased questions (which keyword search misses) are both covered.
- **Cross-encoder reranking** — the top candidates from hybrid search are re-scored by a dedicated reranker model before the best 3 are used, instead of trusting the first-pass retrieval order.
- **Section-aware chunking** — documents are split on their own numbered section headers at ingest time, not blind fixed-size windows, so a heading never gets separated from its own content.
- **Inline source citations** — every factual claim in an answer is cited back to its originating document and section, e.g. `(hr_policy.txt, 2.1 Annual Leave)`.
- **Multi-agent routing** — a LangGraph state machine routes each question to a dedicated Intern / Engineer / Manager agent after retrieval, rather than jamming role logic into one prompt.
- **Self-critique loop** — after an agent answers, a separate critique step checks the answer is actually grounded in the retrieved context and matches the required role style, sending it back for a rewrite (up to 2 retries) if not.
- **Conversation memory** — recent question/answer history carries into follow-up questions ("Is that higher than last year?" resolves correctly), tagged by which role answered so switching roles mid-conversation doesn't leak one role's tone into another's.
- **Adaptive deep rerank** — every 5th question, instead of only reranking what was just fetched, the whole pool of chunks retrieved so far in the conversation gets re-scored against the current question, then pruned back down — keeping long conversations accurate without unbounded context growth.
- **Live activity log** — a sidebar panel showing which agent answered, retrieval counts, and rerank mode per turn, genuinely sourced from the deployed service's own Render logs (not fabricated client-side), scoped to the last hour.

## Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq — Llama 3.3 70B |
| Vector search | FAISS |
| Embeddings + reranking | fastembed (ONNX Runtime — no PyTorch, keeps the whole app under 512MB RAM) |
| Keyword search | BM25 (`rank_bm25`) |
| Backend | FastAPI |
| Frontend | Plain HTML/CSS/JS, no framework |
| Hosting | Render (free tier) |

## Architecture

```
Question + Role
      │
      ▼
 retrieve_node ── hybrid search (FAISS + BM25 → RRF) ── rerank (CrossEncoder)
      │                                                  every 5th turn: rerank the
      │                                                  whole conversation's chunk pool
      ▼
   router (by role)
      │
 ┌────┼────┐
 ▼    ▼    ▼
Intern Eng Manager   ← each generates in its own style, citing sources
 └────┼────┘
      ▼
 critique_node ── grounded in context? matches role style?
      │
   good ──────────────► answer returned
   bad, retries left ──► back to the same role agent, with feedback
   bad, retries used ──► return best attempt anyway
```

## Running locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here

# Optional — enables the live Activity Log panel by reading this app's
# own logs back from Render's API. Leave unset to disable that panel.
RENDER_API_KEY=your_render_api_key_here
RENDER_SERVICE_ID=your_render_service_id_here
```

Build the vector index from the documents in `data/`:

```bash
python ingest.py
```

Start the app:

```bash
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Project layout

```
app/
  main.py        FastAPI app, /ask endpoint
  graph.py        LangGraph state machine (retrieval → routing → generation → critique)
  retriever.py    Hybrid search (FAISS + BM25 + RRF) and reranking
  embeddings.py   ONNX-based embedding adapter for LangChain's FAISS store
  prompts.py      Role-specific prompt templates
  render_logs.py  Fetches this app's own activity from Render's log API
data/             Source documents (HR policy, engineering docs, roadmap, finance)
frontend/         Single-page chat UI
ingest.py         Builds the FAISS index + BM25 corpus from data/
vectorstore/      Generated index (committed — small, and Render has no build step to regenerate it)
```
