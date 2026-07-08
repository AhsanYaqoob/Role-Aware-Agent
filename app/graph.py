import json
import time
from typing import TypedDict

from dotenv import load_dotenv
from groq import APIConnectionError
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.prompts import get_prompt
from app.retriever import get_candidate_texts, rerank_texts

load_dotenv()

RERANK_EVERY_N = 5
POOL_MAX_SIZE = 15
MAX_RETRIES = 2
THREAD_ID = "default-conversation"
CONNECTION_RETRY_ATTEMPTS = 3
CONNECTION_RETRY_BASE_DELAY = 1.0

ROLE_REQUIREMENTS = {
    "intern": "simple, jargon-free language a complete beginner can follow",
    "engineer": "technically detailed and precise, referencing architecture/implementation where relevant",
    "manager": "a maximum of 5 concise bullet points focused on business impact",
}

CRITIQUE_PROMPT = """You are a strict quality reviewer for an AI assistant's answer.

Question: {question}

Retrieved context the answer must be grounded in (not applicable if the question is just a greeting or small talk):
{context}

The answer was written for this role: {role}
Required style for that role: {role_requirement}

Answer to review:
{answer}

Check both:
1. If the question is a real question: is the answer factually grounded in the retrieved context (no invented facts)? If the question is just a greeting or small talk (like "hi", "hello"), this check does not apply -- judge only on whether the reply is warm, brief, and mentions available topics.
2. Does it follow the required style for the role?

Respond with ONLY valid JSON, no other text, in exactly this shape:
{{"verdict": "good", "reason": ""}}
or
{{"verdict": "bad", "reason": "one short sentence on what to fix"}}
"""

llm = ChatGroq(model="llama-3.3-70b-versatile")


def _invoke_llm(prompt: str):
    """Groq calls right after a cold server start can transiently fail to
    resolve DNS; retry a couple of times with backoff before giving up."""
    last_error = None
    for attempt in range(CONNECTION_RETRY_ATTEMPTS):
        try:
            return llm.invoke(prompt)
        except APIConnectionError as exc:
            last_error = exc
            if attempt < CONNECTION_RETRY_ATTEMPTS - 1:
                time.sleep(CONNECTION_RETRY_BASE_DELAY * (attempt + 1))
    raise last_error


class AgentState(TypedDict):
    question: str
    role: str
    context: str
    answer: str
    chunk_pool: list
    query_count: int
    retry_count: int
    feedback: str
    verdict: str


def _role_key(state: AgentState) -> str:
    return state["role"].strip().lower()


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def retrieve_node(state: AgentState) -> dict:
    question = state["question"]
    query_count = state.get("query_count", 0) + 1
    pool = list(state.get("chunk_pool") or [])

    fresh = get_candidate_texts(question, k=10)
    for text in fresh:
        if text not in pool:
            pool.append(text)

    if query_count % RERANK_EVERY_N == 0:
        pool = rerank_texts(question, pool, top_k=POOL_MAX_SIZE)
        context_chunks = pool[:3]
    else:
        context_chunks = rerank_texts(question, fresh, top_k=3)

    return {
        "context": "\n\n".join(context_chunks),
        "chunk_pool": pool,
        "query_count": query_count,
        "retry_count": 0,
        "feedback": "",
        "verdict": "",
    }


def route_by_role(state: AgentState) -> str:
    return _role_key(state)


def _make_role_node(role: str):
    def node(state: AgentState) -> dict:
        prompt = get_prompt(role).format(context=state["context"], question=state["question"])
        if state.get("feedback"):
            prompt += (
                f"\n\nYour previous attempt had this issue: {state['feedback']}. "
                "Please correct it."
            )
        response = _invoke_llm(prompt)
        return {"answer": response.content}

    return node


intern_node = _make_role_node("intern")
engineer_node = _make_role_node("engineer")
manager_node = _make_role_node("manager")


def critique_node(state: AgentState) -> dict:
    prompt = CRITIQUE_PROMPT.format(
        question=state["question"],
        context=state["context"],
        role=state["role"],
        role_requirement=ROLE_REQUIREMENTS[_role_key(state)],
        answer=state["answer"],
    )
    response = _invoke_llm(prompt)

    try:
        result = json.loads(_strip_json_fence(response.content))
        verdict = str(result.get("verdict", "good")).strip().lower()
        reason = result.get("reason", "")
    except (json.JSONDecodeError, AttributeError, TypeError):
        verdict, reason = "good", ""

    retry_count = state.get("retry_count", 0)
    if verdict == "bad":
        retry_count += 1

    return {"verdict": verdict, "feedback": reason, "retry_count": retry_count}


def route_after_critique(state: AgentState) -> str:
    if state["verdict"] == "good":
        return "end"
    if state["retry_count"] > MAX_RETRIES:
        return "end"
    return _role_key(state)


builder = StateGraph(AgentState)
builder.add_node("retrieve_node", retrieve_node)
builder.add_node("intern_node", intern_node)
builder.add_node("engineer_node", engineer_node)
builder.add_node("manager_node", manager_node)
builder.add_node("critique_node", critique_node)

builder.add_edge(START, "retrieve_node")
builder.add_conditional_edges(
    "retrieve_node",
    route_by_role,
    {"intern": "intern_node", "engineer": "engineer_node", "manager": "manager_node"},
)
builder.add_edge("intern_node", "critique_node")
builder.add_edge("engineer_node", "critique_node")
builder.add_edge("manager_node", "critique_node")
builder.add_conditional_edges(
    "critique_node",
    route_after_critique,
    {
        "end": END,
        "intern": "intern_node",
        "engineer": "engineer_node",
        "manager": "manager_node",
    },
)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


def run_graph(question: str, role: str) -> str:
    config = {"configurable": {"thread_id": THREAD_ID}}
    result = graph.invoke({"question": question, "role": role}, config=config)
    return result["answer"]
