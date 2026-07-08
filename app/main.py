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
    answer = run_graph(question, role)
    return JSONResponse({"answer": answer})
