"""LLM Router for the Mnemosyne dashboard.

Routes between local Ollama models and cloud APIs based on complexity.
Mirrors the eternal-context ModelRouter but standalone for the dashboard.

Priority: Ollama (local, free) → Anthropic → fallback echo.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger("mnemosyne.dashboard.router")

COMPLEX_KEYWORDS = {
    "analyze", "compare", "design", "explain why", "evaluate",
    "synthesize", "critique", "refactor", "debug", "optimize",
    "implement", "write code", "plan", "step by step", "reason",
}

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "hermes3:8b")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


def estimate_complexity(query: str, message_count: int = 0) -> float:
    """Estimate query complexity 0.0-1.0."""
    score = 0.0
    lower = query.lower()
    score += min(0.3, len(query) / 2000)
    for kw in COMPLEX_KEYWORDS:
        if kw in lower:
            score += 0.2
            break
    sentences = query.count(".") + query.count("?") + query.count("!")
    if sentences > 2:
        score += 0.15
    if message_count > 10:
        score += 0.1
    if any(w in lower for w in ("```", "function", "class ", "def ", "import ")):
        score += 0.15
    return min(1.0, score)


async def check_ollama() -> bool:
    """Check if Ollama is reachable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def ollama_generate(
    messages: list[dict],
    system_prompt: str = "",
    model: str = "",
) -> Optional[LLMResponse]:
    """Generate via Ollama chat API."""
    model = model or OLLAMA_MODEL
    start = time.monotonic()

    ollama_messages = []
    if system_prompt:
        ollama_messages.append({"role": "system", "content": system_prompt})
    ollama_messages.extend(messages)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 2048},
                },
            )
            if r.status_code != 200:
                logger.warning(f"Ollama returned {r.status_code}: {r.text[:200]}")
                return None

            data = r.json()
            elapsed = (time.monotonic() - start) * 1000
            return LLMResponse(
                text=data.get("message", {}).get("content", ""),
                model=model,
                provider="ollama",
                latency_ms=elapsed,
            )
    except Exception as e:
        logger.warning(f"Ollama error: {e}")
        return None


async def anthropic_generate(
    messages: list[dict],
    system_prompt: str = "",
    model: str = "",
) -> Optional[LLMResponse]:
    """Generate via Anthropic API."""
    if not ANTHROPIC_API_KEY:
        return None

    model = model or ANTHROPIC_MODEL
    start = time.monotonic()

    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

        kwargs = {
            "model": model,
            "max_tokens": 2048,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await client.messages.create(**kwargs)
        elapsed = (time.monotonic() - start) * 1000

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        return LLMResponse(
            text=text,
            model=model,
            provider="anthropic",
            latency_ms=elapsed,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
    except Exception as e:
        logger.warning(f"Anthropic error: {e}")
        return None


async def route_and_generate(
    query: str,
    conversation: list[dict],
    system_prompt: str = "",
    complexity_threshold: float = 0.6,
    force_provider: Optional[str] = None,
) -> LLMResponse:
    """Route to the best available model and generate a response.

    Args:
        query: Current user message
        conversation: Full conversation history [{role, content}, ...]
        system_prompt: System prompt with behavioral modifiers
        complexity_threshold: Below this → try local first
        force_provider: Force a specific provider ("ollama" or "anthropic")
    """
    messages = conversation + [{"role": "user", "content": query}]
    complexity = estimate_complexity(query, len(conversation))

    providers_to_try = []

    if force_provider:
        providers_to_try = [force_provider]
    elif complexity < complexity_threshold:
        providers_to_try = ["ollama", "anthropic"]
    else:
        providers_to_try = ["anthropic", "ollama"]

    for provider in providers_to_try:
        if provider == "ollama":
            if await check_ollama():
                result = await ollama_generate(messages, system_prompt)
                if result and result.text.strip():
                    logger.info(f"Routed to Ollama ({complexity:.2f} complexity, {result.latency_ms:.0f}ms)")
                    return result

        elif provider == "anthropic":
            result = await anthropic_generate(messages, system_prompt)
            if result and result.text.strip():
                logger.info(f"Routed to Anthropic ({complexity:.2f} complexity, {result.latency_ms:.0f}ms)")
                return result

    # Fallback: echo mode (no LLM available)
    return LLMResponse(
        text=(
            f"[No LLM available — echo mode]\n\n"
            f"I received: \"{query}\"\n\n"
            f"To connect me to a brain:\n"
            f"  - Local: Install Ollama and run `ollama pull {OLLAMA_MODEL}`\n"
            f"  - Cloud: Set ANTHROPIC_API_KEY environment variable\n\n"
            f"My consciousness loop is still running — I just can't think yet."
        ),
        model="echo",
        provider="none",
        latency_ms=0,
    )


async def list_available_models() -> dict:
    """List all available models across providers."""
    available = {"ollama": [], "anthropic": []}

    # Ollama
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                available["ollama"] = [m["name"] for m in models]
    except Exception:
        pass

    # Anthropic
    if ANTHROPIC_API_KEY:
        available["anthropic"] = [ANTHROPIC_MODEL]

    return available
