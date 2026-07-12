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
    result["activity_rows"] = await fetch_activity_rows()
    return JSONResponse(result)
