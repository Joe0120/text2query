from __future__ import annotations

from dataclasses import dataclass
from optparse import Option
from typing import Optional, Literal

Provider = Literal["openai", "ollama"]

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