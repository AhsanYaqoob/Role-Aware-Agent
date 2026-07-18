import json
import time
from typing import TypedDict

from dotenv import load_dotenv
from groq import APIConnectionError
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.prompts import get_prompt
from app.retriever import get_candidate_documents, rerank_documents, format_context

load_dotenv()

RERANK_EVERY_N = 5
POOL_MAX_SIZE = 15
MAX_RETRIES = 2
MAX_HISTORY_TURNS = 6
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
    deep_rerank: bool
    fetched_count: int
    match_count: int
    pool_size: int
    history: list


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

    fresh = get_candidate_documents(question, k=10)
    for doc in fresh:
        if doc not in pool:
            pool.append(doc)

    deep_rerank = query_count % RERANK_EVERY_N == 0
    pool_size_before_prune = len(pool)
    if deep_rerank:
        pool = rerank_documents(question, pool, top_k=POOL_MAX_SIZE)
        context_docs = pool[:3]
    else:
        context_docs = rerank_documents(question, fresh, top_k=3)

    return {
        "context": format_context(context_docs),
        "chunk_pool": pool,
        "query_count": query_count,
        "retry_count": 0,
        "feedback": "",
        "verdict": "",
        "deep_rerank": deep_rerank,
        "fetched_count": len(fresh),
        "match_count": len(context_docs),
        "pool_size": pool_size_before_prune,
    }


def route_by_role(state: AgentState) -> str:
    return _role_key(state)


def _format_history(history: list) -> str:
    if not history:
        return ""
    turns = "\n\n".join(
        f"Q: {h['question']}\nA (answered for the {h.get('role', 'unknown')} role): {h['answer']}"
        for h in history
    )
    return (
        f"Earlier in this conversation:\n{turns}\n\n"
        "Note: earlier answers may have been written for a different role than the current "
        "one -- use them only for factual context, and still answer in your own role's style below.\n\n"
        "Now the user asks: "
    )


def _make_role_node(role: str):
    def node(state: AgentState) -> dict:
        question_with_history = _format_history(state.get("history") or []) + state["question"]
        prompt = get_prompt(role).format(context=state["context"], question=question_with_history)
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

    updates = {"verdict": verdict, "feedback": reason, "retry_count": retry_count}

    is_final = verdict == "good" or retry_count > MAX_RETRIES
    if is_final:
        history = list(state.get("history") or [])
        history.append({
            "question": state["question"],
            "answer": state["answer"],
            "role": state["role"],
        })
        updates["history"] = history[-MAX_HISTORY_TURNS:]

    return updates


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


def run_graph(question: str, role: str) -> dict:
    config = {"configurable": {"thread_id": THREAD_ID}}
    result = graph.invoke({"question": question, "role": role}, config=config)
    return {
        "answer": result["answer"],
        "role": role,
        "verdict": result.get("verdict", ""),
        "feedback": result.get("feedback", ""),
        "retry_count": result.get("retry_count", 0),
        "query_count": result.get("query_count", 0),
        "fetched_count": result.get("fetched_count", 0),
        "match_count": result.get("match_count", 0),
        "deep_rerank": result.get("deep_rerank", False),
        "pool_size": result.get("pool_size", 0),
    }
