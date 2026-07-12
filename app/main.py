from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.graph import run_graph

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="frontend")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


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
    return JSONResponse(result)
