from .model_configs import ModelConfig
import litellm
from typing import List, Optional, Dict, Any
import asyncio

# Configure litellm to be quiet if needed
# litellm.suppress_debug_info = True

def build_model_name(config: ModelConfig) -> str:
    """
    Build the model name string for litellm.
    Example: "azure/gpt-4", "vertex_ai/gemini-pro", etc.
    """
    if config.provider == "openai":
        return config.model_name
    return f"{config.provider}/{config.model_name}"

async def agenerate_chat(config: ModelConfig, messages: List[Dict[str, str]]) -> str:
    """
    Generate a chat response from a model using litellm.
    """
    model = build_model_name(config)
    
    # Common arguments for litellm.completion
    kwargs = {
        "model": model,
        "messages": messages,
        "timeout": config.timeout,
        "temperature": config.temperature,
    }
    
    # Only pass api_base/key if provided, allowing litellm to fallback to env vars
    if config.base_url:
        kwargs["api_base"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
        
    # Add optional max_tokens if set
    if config.max_tokens:
        kwargs["max_tokens"] = config.max_tokens

    # Apply extra provider-specific kwargs
    if config.extra_kwargs:
        kwargs.update(config.extra_kwargs)

    try:
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LiteLLM completion failed for {model}: {str(e)}") from e

async def aembed_text(config: ModelConfig, text: str) -> List[float]:
    """
    Generate text embeddings using litellm.
    """
    model = build_model_name(config)
    
    # Common arguments
    kwargs = {
        "model": model,
        "input": [text],
        "timeout": config.timeout
    }
    
    if config.base_url:
        kwargs["api_base"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
        
    if config.extra_kwargs:
        kwargs.update(config.extra_kwargs)
        
    try:
        response = await litellm.aembedding(**kwargs)
        return response.data[0]["embedding"]
    except Exception as e:
        raise RuntimeError(f"LiteLLM embedding failed for {model}: {str(e)}") from e
