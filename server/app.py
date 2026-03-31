"""
thinkrouter.server.app
~~~~~~~~~~~~~~~~~~~~~~
FastAPI proxy server — drop-in OpenAI-compatible endpoint.

Instead of pointing your code at api.openai.com, point it at this server.
ThinkRouter routes every request automatically.

Start:
    pip install thinkrouter[server]
    uvicorn thinkrouter.server.app:app --host 0.0.0.0 --port 8000

Then in your code — just change the base_url:
    from openai import OpenAI
    client = OpenAI(
        api_key="your-openai-key",
        base_url="http://localhost:8000/v1",
    )
    # Every call is now routed by ThinkRouter automatically.
"""
from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "Server requires:  pip install thinkrouter[server]"
    ) from exc

from thinkrouter import ThinkRouter
from thinkrouter.constants import OPENAI_REASONING_EFFORT, OPENAI_REASONING_MODELS, Tier

app = FastAPI(
    title="ThinkRouter Proxy",
    description="OpenAI-compatible API proxy with automatic query difficulty routing.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global router instance (initialised on first request using the caller's key)
_routers: Dict[str, ThinkRouter] = {}


def _get_router(api_key: str, provider: str, model: str) -> ThinkRouter:
    key = f"{provider}:{api_key[:8]}:{model}"
    if key not in _routers:
        _routers[key] = ThinkRouter(
            provider=provider,
            api_key=api_key,
            model=model,
            verbose=os.getenv("THINKROUTER_VERBOSE", "").lower() in ("1", "true"),
        )
    return _routers[key]


# ── Request / Response models ─────────────────────────────────────────────────

class Message(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    model:       str
    messages:    List[Message]
    temperature: float = 0.7
    stream:      bool  = False
    max_tokens:  Optional[int] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o",      "object": "model"},
            {"id": "gpt-4o-mini", "object": "model"},
            {"id": "o1",          "object": "model"},
            {"id": "o3-mini",     "object": "model"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatRequest):
    # Extract API key from Authorization header
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    api_key = auth.removeprefix("Bearer ").strip()

    router = _get_router(api_key, "openai", body.model)

    # Use the last user message as the query for classification
    user_messages = [m for m in body.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in request")

    query    = user_messages[-1].content
    clf      = router.classify(query)
    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    if body.stream:
        async def token_stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in router.astream(query=query, model=body.model):
                    data = (
                        'data: {"id":"chatcmpl-tr","object":"chat.completion.chunk",'
                        f'"choices":[{{"delta":{{"content":{chunk!r}}},"index":0}}]}}\n\n'
                    )
                    yield data.encode()
                yield b"data: [DONE]\n\n"
            except Exception as exc:
                yield f"data: {{\"error\": \"{exc}\"}}\n\n".encode()

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    response = await router.achat(
        query=query, model=body.model, messages=messages,
        temperature=body.temperature,
    )

    return {
        "id":      "chatcmpl-tr",
        "object":  "chat.completion",
        "model":   body.model,
        "choices": [
            {
                "index":         0,
                "message":       {"role": "assistant", "content": response.content},
                "finish_reason": "stop",
            }
        ],
        "usage": response.usage_tokens,
        "thinkrouter": {
            "tier":             response.routing.tier.name,
            "confidence":       round(response.routing.confidence, 4),
            "token_budget":     response.routing.token_budget,
            "classifier_ms":    round(response.routing.latency_ms, 3),
            "reasoning_effort": response.reasoning_effort,
            "thinking_budget":  response.thinking_budget,
        },
    }


@app.get("/v1/usage")
async def usage_summary():
    """Return aggregate usage statistics across all routers."""
    summaries = []
    for key, router in _routers.items():
        s = router.usage.summary()
        summaries.append({
            "router":        key,
            "total_calls":   s.total_calls,
            "tokens_saved":  s.total_tokens_saved,
            "savings_pct":   round(s.savings_pct, 2),
            "avg_latency_ms": round(s.avg_latency_ms, 3),
        })
    return {"routers": summaries}
