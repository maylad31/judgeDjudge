from openai import OpenAI
from typing import Type, Any, Optional, Dict
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


def call_openrouter(
    prompt: str,
    response_model: Type[BaseModel],
    model: str | None = None,
    openai_params: Optional[Dict[str, Any]] = None,
    provider_params: Optional[Dict[str, Any]] = None,
) -> BaseModel | Any:
    """
    Call OpenRouter using the OpenAI Responses API with Pydantic-based structured outputs.

    Args:
        prompt: The user prompt
        response_model: Pydantic model class for structured output
        model: Model identifier (e.g., "openai/gpt-4o-mini")
        openai_params: Dict of OpenAI-compatible parameters (e.g., {"temperature": 0.7})
        provider_params: Dict of OpenRouter-specific parameters (e.g., {"zdr": True})

    Returns:
        Pydantic model instance (or response object if streaming)
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    openai_params = openai_params or {}
    provider_params = provider_params or {}
    messages = [{"role": "user", "content": prompt}]

    params = {
        "model": model,
        "input": messages,
        "text_format": response_model,
    }
    if openai_params:
        params.update(openai_params)

    response = client.responses.parse(
        **params,
        extra_body=provider_params if provider_params else None,
    )

    return response.output_parsed
