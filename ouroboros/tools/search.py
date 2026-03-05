"""Web search tool — uses OpenRouter (Perplexity) instead of OpenAI."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry


def _web_search(ctx: ToolContext, query: str) -> str:
    """Search via OpenRouter using a web-connected model (Perplexity sonar)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "OPENROUTER_API_KEY not set; web_search unavailable."})

    try:
        import httpx

        model = os.environ.get(
            "OUROBOROS_WEBSEARCH_MODEL", "perplexity/sonar"
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/shanpenghui/ouroboros",
            "X-Title": "Ouroboros",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ],
        }
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "(no answer)")
        )

        # Extract citations if present (Perplexity returns them)
        citations = data.get("citations", [])
        result: Dict[str, Any] = {"answer": answer}
        if citations:
            result["sources"] = citations

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": repr(e)}, ensure_ascii=False)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            "web_search",
            {
                "name": "web_search",
                "description": (
                    "Search the web via OpenRouter (Perplexity sonar). "
                    "Returns JSON with answer + sources."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
            _web_search,
        ),
    ]
