from .model_configs import ModelConfig
import requests
from typing import List, Optional, Dict, Any
import asyncio

def build_url(config: ModelConfig) -> str:
    return f"{config.base_url.rstrip('/')}/{config.endpoint.lstrip('/')}"

def build_header(config: ModelConfig) -> Dict[str, str]:
    header: Dict[str, str] = {
        "Content-Type": "application/json",
    }
    # Groq and OpenAI both use Bearer token authentication
    if config.provider in ("openai", "groq"):
        header["Authorization"] = f"Bearer {config.api_key}"
    return header

def parse_response(config: ModelConfig, data: Dict[str, Any]) -> str:
    # Groq uses OpenAI-compatible response format
    if config.provider in ("openai", "groq"):
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected {config.provider} chat response format: {data}") from e

    if config.provider == "ollama":
        try:
            return data["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected Ollama chat response format: {data}") from e

    raise ValueError(f"Unsupported provider in parse_response: {config.provider}")

def generate_chat(config: ModelConfig, messages: List[Dict[str, Any]]) -> str:
    """
    Generate a chat response from a model
    """
    url = build_url(config)
    header = build_header(config)
    payload = {
        "model": config.model_name,
        "messages": messages,
    }
    if config.max_tokens:
        payload["max_tokens"] = config.max_tokens
    if config.temperature:
        payload["temperature"] = config.temperature

    response = requests.post(url, headers=header, timeout=config.timeout, json=payload)
    response.raise_for_status()
    data = response.json()

    return parse_response(config, data)

async def agenerate_chat(config: ModelConfig, messages: List[Dict[str, str]]) -> str:
    return await asyncio.to_thread(generate_chat, config, messages)

def parse_embedding(config: ModelConfig, data: Dict[str, Any]) -> List[float]:
    # Groq uses OpenAI-compatible embedding response format
    if config.provider in ("openai", "groq"):
        return data["data"][0]["embedding"]
    elif config.provider == "ollama":
        return data["embeddings"][0]
    raise ValueError(f"Unsupported provider in parse_embedding: {config.provider}")

def embed_text(config: ModelConfig, text: str) -> List[float]:
    url = build_url(config)
    header = build_header(config)
    payload = {
        "model": config.model_name,
        "input": [text],
    }
    response = requests.post(url, headers=header, timeout=config.timeout, json=payload)
    response.raise_for_status()
    data = response.json()
    return parse_embedding(config, data)

async def aembed_text(config: ModelConfig, text: str) -> List[float]:
    return await asyncio.to_thread(embed_text, config, text)
