"""
thinkrouter.server.app
~~~~~~~~~~~~~~~~~~~~~~
Production-ready OpenAI-compatible proxy server.

Point any existing OpenAI SDK at this server — routing is automatic.

    # Start the server
    pip install thinkrouter[server]
    uvicorn thinkrouter.server.app:app --host 0.0.0.0 --port 8000

    # In your application — change only base_url
    from openai import OpenAI
    client = OpenAI(
        api_key="your-openai-key",
        base_url="http://localhost:8000/v1",
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is 2+3?"}]
    )
    # Automatically routed to NO_THINK → 50 tokens used
    # response.thinkrouter shows routing metadata

Environment variables:
    OPENAI_API_KEY      / ANTHROPIC_API_KEY
    THINKROUTER_VERBOSE  1 | 0
    THINKROUTER_BACKEND  heuristic | distilbert
    THINKROUTER_THRESHOLD  float (default 0.75)
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        "Server requires:  pip install thinkrouter[server]"
    ) from exc

from thinkrouter import ThinkRouter
from thinkrouter.config import Config
from thinkrouter.constants import OPENAI_REASONING_MODELS
from thinkrouter.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)

# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ThinkRouter Proxy",
    description=(
        "OpenAI-compatible proxy with automatic query difficulty routing. "
        "Reduces LLM reasoning-token costs by 50-60% transparently."
    ),
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Router cache: one ThinkRouter per (provider, api_key_prefix, model) ──

_ROUTERS: Dict[str, ThinkRouter] = {}


def _get_router(api_key: str, model: str) -> ThinkRouter:
    key = f"openai:{api_key[:12]}:{model}"
    if key not in _ROUTERS:
        cfg = Config()
        _ROUTERS[key] = ThinkRouter(
            provider="openai",
            api_key=api_key,
            model=model,
            classifier_backend=cfg.classifier_backend,
            confidence_threshold=cfg.confidence_threshold,
            max_retries=cfg.max_retries,
            verbose=cfg.verbose,
        )
    return _ROUTERS[key]


# ── Request / Response models ─────────────────────────────────────────────

class MessageIn(BaseModel):
    role:    str
    content: str


class ChatCompletionRequest(BaseModel):
    model:       str
    messages:    List[MessageIn]
    temperature: float         = Field(default=0.7, ge=0.0, le=2.0)
    stream:      bool          = False
    max_tokens:  Optional[int] = None


# ── Exception handlers ────────────────────────────────────────────────────

@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError):
    return JSONResponse(status_code=429, content={"error": {"message": str(exc), "type": "rate_limit_error"}})


@app.exception_handler(AuthenticationError)
async def auth_handler(request: Request, exc: AuthenticationError):
    return JSONResponse(status_code=401, content={"error": {"message": str(exc), "type": "authentication_error"}})


@app.exception_handler(ModelNotFoundError)
async def model_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(status_code=404, content={"error": {"message": str(exc), "type": "model_not_found"}})


@app.exception_handler(ProviderError)
async def provider_handler(request: Request, exc: ProviderError):
    code = exc.status_code or 500
    return JSONResponse(status_code=code, content={"error": {"message": str(exc), "type": "provider_error"}})


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.3.0", "service": "thinkrouter-proxy"}


@app.get("/v1/models", tags=["models"])
async def list_models():
    """List available models (mirrors OpenAI response format)."""
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "thinkrouter"}
            for m in [
                "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
                "o1", "o1-mini", "o3", "o3-mini", "o4-mini",
            ]
        ],
    }


@app.post("/v1/chat/completions", tags=["chat"])
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with automatic routing.

    The response includes a `thinkrouter` metadata field showing
    the routing decision applied.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <key>")
    api_key = auth.removeprefix("Bearer ").strip()

    router = _get_router(api_key, body.model)

    # Use the last user message as the classification query
    user_msgs = [m for m in body.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message in request")

    query    = user_msgs[-1].content
    clf      = router.classify(query)
    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    if body.stream:
        async def _token_stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in router.astream(query=query, model=body.model):
                    payload = {
                        "id":      "chatcmpl-tr",
                        "object":  "chat.completion.chunk",
                        "model":   body.model,
                        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            except ProviderError as exc:
                err = json.dumps({"error": str(exc)})
                yield f"data: {err}\n\n".encode()

        return StreamingResponse(_token_stream(), media_type="text/event-stream")

    response = await router.achat(
        query=query, model=body.model,
        messages=messages, temperature=body.temperature,
    )

    return {
        "id":      "chatcmpl-tr",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   body.model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": response.content},
            "finish_reason": "stop",
        }],
        "usage": response.usage_tokens,
        "thinkrouter": {
            "tier":             response.routing.tier.name,
            "tier_label":       response.routing.tier.name.lower(),
            "confidence":       round(response.routing.confidence, 4),
            "token_budget":     response.routing.token_budget,
            "classifier_ms":    round(response.routing.latency_ms, 3),
            "reasoning_effort": response.reasoning_effort,
            "thinking_budget":  response.thinking_budget,
        },
    }


@app.get("/v1/usage", tags=["system"])
async def usage_stats():
    """Aggregate usage statistics across all active routers."""
    result = []
    for cache_key, router in _ROUTERS.items():
        s = router.usage.summary()
        result.append({
            "router":          cache_key,
            "total_calls":     s.total_calls,
            "tokens_saved":    s.total_tokens_saved,
            "savings_pct":     round(s.savings_pct, 2),
            "avg_latency_ms":  round(s.avg_latency_ms, 3),
            "tier_breakdown":  {
                k.name: v for k, v in s.tier_breakdown.items()
            },
        })
    return {"routers": result, "total_routers": len(result)}
