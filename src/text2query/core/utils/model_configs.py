from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

Provider = Literal["openai", "ollama", "azure", "vertex_ai", "bedrock", "groq", "anthropic"] | str

@dataclass
class ModelConfig:
    base_url: str
    endpoint: str
    api_key: str
    model_name: str
    provider: Provider = "openai"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.2
    timeout: int = 30
    extra_kwargs: Optional[Dict[str, Any]] = None


def create_model_config(
    model_name: str,
    api_base: str = "",
    apikey: str = "",
    endpoint: Optional[str] = None,
    provider: Provider = "openai",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = 0.2,
    timeout: int = 30,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> ModelConfig:
    """
    Create a ModelConfig instance with simplified parameter names.
    
    This is a convenience function that maps user-friendly parameter names
    (api_base, apikey) to the ModelConfig dataclass fields (base_url, api_key).
    
    Args:
        model_name: Name of the LLM model to use
        api_base: Base URL for the API (e.g., "https://api.openai.com" or "http://localhost:11434")
        apikey: API key for authentication
        endpoint: API endpoint path (defaults to "/v1/chat/completions" for openai, "/api/chat" for ollama)
        provider: LLM provider type ("openai" or "ollama", default: "openai")
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (default: 0.2)
        timeout: Request timeout in seconds (default: 30)
    
    Returns:
        ModelConfig: Configured ModelConfig instance
    
    Example:
        >>> from text2query.core.utils.model_configs import create_model_config
        >>> 
        >>> # For OpenAI
        >>> llm_config = create_model_config(
        ...     model_name="gpt-4",
        ...     api_base="https://api.openai.com",
        ...     apikey="sk-...",
        ...     provider="openai"
        ... )
        >>> 
        >>> # For Ollama
        >>> llm_config = create_model_config(
        ...     model_name="llama2",
        ...     api_base="http://localhost:11434",
        ...     apikey="",
        ...     provider="ollama"
        ... )
    """
    # Set default endpoints based on provider
    # Auto-fix for OpenAI api_base if it is standard URL but missing /v1
    if provider == "openai" and api_base:
        if "api.openai.com" in api_base and not api_base.endswith("/v1") and not api_base.endswith("/v1/"):
            api_base = api_base.rstrip("/") + "/v1"

    return ModelConfig(
        base_url=api_base,
        endpoint=endpoint,
        api_key=apikey,
        model_name=model_name,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        extra_kwargs=extra_kwargs or {},
    )
create_llm_config = create_model_config
