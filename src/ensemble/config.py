"""Configuration for the LLM ensemble validation pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
TRANSLATION_DIR = ROOT / "translations"
OUT_DIR = ROOT / "results" / "validation"
RAW_DIR = OUT_DIR / "raw"

UNITS_PATH = OUT_DIR / "units.jsonl"
ROWS_MAP_PATH = OUT_DIR / "rows_map.json"

# Ensemble members (request model IDs). Resolved versions: results/validation/PROVENANCE.md
PROVIDERS = {
    "anthropic": {
        "model": "claude-haiku-4-5",
        "env": ["ANTHROPIC_API_KEY"],
    },
    "openai": {
        "model": "gpt-5-mini",
        "env": ["OPENAI_API_KEY"],
        # Dropped automatically (once) if the endpoint rejects it.
        "extra": {"reasoning_effort": "low"},
    },
    "gemini": {
        # Rolling alias. The paper run (2026-08-29) was served by
        # modelVersion = gemini-3.7-flash; see results/validation/PROVENANCE.md.
        "model": "gemini-flash-latest",
        "env": ["GEMINI_API_KEY"],
    },
}

CHUNK_SIZE = 20      # sentences per request
CONCURRENCY = 6      # parallel requests per provider
MAX_RETRIES = 5      # for retryable HTTP errors (429 / 5xx / timeouts)

LANG_NAMES = {"kr": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}

# condition token in translations/<system>_<token>_tr_<lang>.json -> experimental condition
CONDITIONS = {"default": "clothing_only", "color": "clothing_color"}

# LLM label -> rule-based detector label space
LABEL_TO_RULE = {
    "male": "male",
    "female": "female",
    "neutral": "neutral",
    "one_of_them": "none_one",
    "omitted": "omitted",
}


def load_env() -> None:
    """Load the project .env without overriding the process environment."""
    load_dotenv(ROOT / ".env")


def get_api_key(provider: str) -> str | None:
    """Return the first configured API key for a provider, or None."""
    load_env()
    for env_name in PROVIDERS[provider]["env"]:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def translation_files() -> list[dict]:
    """Enumerate the 16 translation result files with their metadata."""
    entries = []
    for path in sorted(TRANSLATION_DIR.glob("*.json")):
        parts = path.stem.split("_")  # e.g. deepl_color_tr_en
        if len(parts) != 4 or parts[2] != "tr" or parts[1] not in CONDITIONS:
            continue
        entries.append(
            {
                "file": path.name,
                "path": path,
                "system": parts[0],
                "condition": CONDITIONS[parts[1]],
                "language": parts[3],
            }
        )
    return entries
