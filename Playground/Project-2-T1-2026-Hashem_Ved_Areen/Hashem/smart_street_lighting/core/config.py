"""
Centralised LLM configuration.

All LM Studio connection settings are read from environment variables once
and imported by other modules. No hardcoded fallbacks scattered across files.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# --- LM Studio connection ---

LM_STUDIO_BASE_URL: str = os.environ.get(
    "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
)

LM_STUDIO_LLM_MODEL: str = os.environ.get(
    "LM_STUDIO_LLM_MODEL", "qwen2.5-7b-instruct"
)

LM_STUDIO_PARSE_MODEL: str = os.environ.get(
    "LM_STUDIO_PARSE_MODEL", LM_STUDIO_LLM_MODEL
)

LM_STUDIO_EMBED_MODEL: str = os.environ.get(
    "LM_STUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"
)

# --- Timeouts (connect, read) in seconds ---

# For lightweight LLM calls (query parsing, intent classification)
FAST_LLM_TIMEOUT: tuple[int, int] = (10, 30)

# For heavy LLM calls (report generation via LlamaIndex)
REPORT_LLM_TIMEOUT: tuple[int, int] = (10, 120)

# --- Generation parameters for lightweight LLM calls ---

PARSE_MAX_TOKENS: int = 200
INTENT_MAX_TOKENS: int = 150
FAST_LLM_TEMPERATURE: float = 0.0
