from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.graph import run_graph
from app.render_logs import fetch_activity_rows

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="frontend")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools_probe():
    # Chrome DevTools requests this automatically whenever it's open, probing
    # for optional project settings. Respond with an empty object instead of
    # a 404 so it doesn't show up as noise in the logs.
    return JSONResponse({})


@app.post("/ask")
async def ask(question: str = Form(...), role: str = Form(...)):
    result = run_graph(question, role)
    print(
        f"[activity] turn={result['query_count']} role={result['role']} "
        f"verdict={result['verdict'] or 'good'} fetched={result['fetched_count']} "
        f"matched={result['match_count']} "
        f"rerank={'deep' if result['deep_rerank'] else 'normal'} "
        f"retries={result['retry_count']}"
    )

    current_row = {
        "query_count": result["query_count"],
        "role": result["role"],
        "verdict": result["verdict"] or "good",
        "fetched_count": result["fetched_count"],
        "match_count": result["match_count"],
        "deep_rerank": result["deep_rerank"],
        "retry_count": result["retry_count"],
        "feedback": result["feedback"],
    }

    # Render's log search has real ingestion lag -- the line we just printed
    # above usually isn't searchable yet by the time we'd query it here, so
    # this turn's own row would be missing from the table for one full
    # request. Show it immediately from data we already computed, and let
    # Render's log history (fetched below) fill in everything before it.
    history_rows = await fetch_activity_rows()
    if history_rows and history_rows[0]["query_count"] == current_row["query_count"] and history_rows[0]["role"] == current_row["role"]:
        history_rows = history_rows[1:]  # Render already ingested this turn -- don't show it twice
    result["activity_rows"] = [current_row] + history_rows

    return JSONResponse(result)
