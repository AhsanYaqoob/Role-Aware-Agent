import os
import re
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

RENDER_API_KEY = os.getenv("RENDER_API_KEY", "")
RENDER_SERVICE_ID = os.getenv("RENDER_SERVICE_ID", "")
RENDER_SERVICE_NAME = os.getenv("RENDER_SERVICE_NAME", "")
RENDER_API_BASE = "https://api.render.com/v1"

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_ACTIVITY_LINE_RE = re.compile(
    r"\[activity\] turn=(\d+) role=(\w+) verdict=(\w+) fetched=(\d+) "
    r"matched=(\d+) rerank=(\w+) retries=(\d+)"
)

_service_cache = {"id": None, "owner_id": None, "resolved_at": 0.0}
_SERVICE_CACHE_TTL = 3600

_rows_cache = {"data": [], "fetched_at": 0.0}
_ROWS_CACHE_TTL = 3.0


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _unwrap(item):
    return item.get("service", item) if isinstance(item, dict) else item


async def _resolve_service(client: httpx.AsyncClient) -> tuple[str, str]:
    now = time.time()
    if _service_cache["id"] and now - _service_cache["resolved_at"] < _SERVICE_CACHE_TTL:
        return _service_cache["id"], _service_cache["owner_id"]

    if RENDER_SERVICE_ID:
        resp = await client.get(
            f"{RENDER_API_BASE}/services/{RENDER_SERVICE_ID}",
            headers={"Authorization": f"Bearer {RENDER_API_KEY}"},
        )
        resp.raise_for_status()
        service = _unwrap(resp.json())
    else:
        resp = await client.get(
            f"{RENDER_API_BASE}/services",
            headers={"Authorization": f"Bearer {RENDER_API_KEY}"},
            params={"name": RENDER_SERVICE_NAME, "limit": 1},
        )
        resp.raise_for_status()
        items = resp.json()
        if not items:
            raise ValueError(f"No Render service found matching name '{RENDER_SERVICE_NAME}'")
        service = _unwrap(items[0])

    service_id = service["id"]
    owner_id = service["ownerId"]
    _service_cache.update(id=service_id, owner_id=owner_id, resolved_at=now)
    return service_id, owner_id


async def fetch_activity_rows(limit: int = 20) -> list[dict]:
    """Fetches recent log lines from this app's own Render service and
    parses out ONLY the '[activity] ...' lines this app itself prints.
    Never returns raw log text -- just the same safe, already-public fields
    (role/verdict/counts) reconstructed from Render's log stream, which is
    the actual source of truth here. Returns [] on any failure or if
    RENDER_API_KEY isn't configured; never raises."""
    if not RENDER_API_KEY:
        return []

    now = time.time()
    if now - _rows_cache["fetched_at"] < _ROWS_CACHE_TTL:
        return _rows_cache["data"]

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            service_id, owner_id = await _resolve_service(client)
            resp = await client.get(
                f"{RENDER_API_BASE}/logs",
                headers={"Authorization": f"Bearer {RENDER_API_KEY}"},
                params={
                    "ownerId": owner_id,
                    "resource": [service_id],
                    "text": ["[activity]"],
                    "limit": limit,
                    "direction": "backward",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("logs", data) if isinstance(data, dict) else data

            rows = []
            for entry in entries:
                message = entry.get("message", "") if isinstance(entry, dict) else str(entry)
                match = _ACTIVITY_LINE_RE.search(_strip_ansi(message))
                if not match:
                    continue
                turn, role, verdict, fetched, matched, rerank, retries = match.groups()
                # "turn" (query_count) resets to 1 on every server restart, so it
                # is NOT reliable for ordering across restarts -- an old session's
                # turn=6 can be chronologically older than a fresh session's
                # turn=1. Render's own log timestamp is the real clock.
                timestamp = entry.get("timestamp", "") if isinstance(entry, dict) else ""
                rows.append({
                    "query_count": int(turn),
                    "role": role,
                    "verdict": verdict,
                    "fetched_count": int(fetched),
                    "match_count": int(matched),
                    "deep_rerank": rerank == "deep",
                    "retry_count": int(retries),
                    "feedback": "",
                    "_timestamp": timestamp,
                })

            # ISO 8601 timestamps sort correctly as plain strings.
            rows.sort(key=lambda r: r["_timestamp"], reverse=True)
            for row in rows:
                del row["_timestamp"]
            _rows_cache.update(data=rows, fetched_at=now)
            return rows
    except Exception:
        return []
