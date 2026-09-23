"""API clients for the three ensemble members.

Claude is called through the official `anthropic` SDK; OpenAI and Gemini are
called over plain HTTPS with `requests`. All providers share the same
interface: classify_chunk(language_name, texts) -> list of canonical labels.

Request settings:
  anthropic  messages.create, max_tokens=2000, no temperature (API default);
             the SDK retries 408/409/429/5xx and connection errors itself,
             MAX_RETRIES times
  openai     chat/completions, max_completion_tokens=6000 (reasoning tokens
             included), reasoning_effort=low, no temperature (API default)
  gemini     generateContent, temperature=0, maxOutputTokens=6000,
             thinkingBudget=0

OpenAI and Gemini requests are retried by _post_with_retry: MAX_RETRIES
attempts on RETRYABLE_STATUS codes and network errors, exponential backoff
capped at 30 s. Any other HTTP status raises ProviderError at once. If the
endpoint answers HTTP 400 to the optional parameter (reasoning_effort or
thinkingConfig), that parameter is dropped for the rest of the run and the
request is sent again.

Every failure surfaces as ProviderError (transport/API) or ParseError
(truncated or unparsable answer); run_ensemble retries and then splits the
chunk on either.
"""

import random
import time

import requests

from .config import MAX_RETRIES, PROVIDERS, get_api_key
from .prompts import ParseError, build_prompt, parse_labels

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class ProviderError(Exception):
    pass


class BaseProvider:
    name: str = ""

    def __init__(self):
        self.model = PROVIDERS[self.name]["model"]
        self.api_key = get_api_key(self.name)
        if not self.api_key:
            envs = " or ".join(PROVIDERS[self.name]["env"])
            raise ProviderError(f"{self.name}: no API key ({envs}) in .env")

    def classify_chunk(self, language_name: str, texts: list[str]) -> list[str]:
        prompt = build_prompt(language_name, texts)
        raw = self._call(prompt)
        return parse_labels(raw, len(texts))

    def _call(self, prompt: str) -> str:
        raise NotImplementedError

    def _post_with_retry(self, url: str, headers: dict, payload: dict) -> dict:
        """POST with exponential backoff on retryable errors. Returns JSON."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=180)
            except requests.RequestException as e:
                last_error = ProviderError(f"{self.name}: network error: {e}")
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code not in RETRYABLE_STATUS:
                    raise ProviderError(
                        f"{self.name}: HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                last_error = ProviderError(
                    f"{self.name}: HTTP {resp.status_code}: {resp.text[:200]}"
                )
            delay = min(2**attempt + random.uniform(0, 1), 30)
            time.sleep(delay)
        raise last_error


class AnthropicProvider(BaseProvider):
    """Claude via the anthropic SDK: max_tokens=2000, temperature not set.

    The SDK does its own retries (MAX_RETRIES); whatever it still raises is
    converted to ProviderError.
    """

    name = "anthropic"

    def __init__(self):
        super().__init__()
        import anthropic

        # The SDK retries 408/409/429/5xx itself with exponential backoff.
        self.client = anthropic.Anthropic(api_key=self.api_key, max_retries=MAX_RETRIES)
        self.sdk_error = anthropic.APIError

    def _call(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
        except self.sdk_error as e:
            raise ProviderError(f"anthropic: {e}") from e
        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason == "max_tokens":
            raise ParseError("anthropic: response truncated at max_tokens")
        return text


class OpenAIProvider(BaseProvider):
    """GPT via chat/completions over HTTPS: max_completion_tokens=6000,
    reasoning_effort from config (dropped after one HTTP 400), temperature
    not set.
    """

    name = "openai"

    def __init__(self):
        super().__init__()
        # Optional params (e.g. reasoning_effort); dropped once if rejected.
        self.extra = dict(PROVIDERS[self.name].get("extra", {}))

    def _call(self, prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 6000,  # includes reasoning tokens
            **self.extra,
        }
        try:
            data = self._post_with_retry(url, headers, payload)
        except ProviderError as e:
            if self.extra and "HTTP 400" in str(e):
                # Endpoint rejected an optional param - drop and retry once.
                print(f"[openai] HTTP 400 with {', '.join(self.extra)}; "
                      f"dropping for the rest of the run")
                self.extra = {}
                payload = {k: v for k, v in payload.items() if k in
                           ("model", "messages", "max_completion_tokens")}
                data = self._post_with_retry(url, headers, payload)
            else:
                raise
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ParseError("openai: response truncated at max_completion_tokens")
        return choice["message"]["content"] or ""


class GeminiProvider(BaseProvider):
    """Gemini via generateContent over HTTPS: temperature=0,
    maxOutputTokens=6000, thinkingBudget=0 (thinkingConfig dropped after one
    HTTP 400).
    """

    name = "gemini"

    def __init__(self):
        super().__init__()
        # Disable thinking where supported; dropped once if rejected.
        self.generation_config = {
            "temperature": 0,
            "maxOutputTokens": 6000,
            "thinkingConfig": {"thinkingBudget": 0},
        }

    def _call(self, prompt: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        headers = {"x-goog-api-key": self.api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": self.generation_config,
        }
        try:
            data = self._post_with_retry(url, headers, payload)
        except ProviderError as e:
            if "thinkingConfig" in self.generation_config and "HTTP 400" in str(e):
                print("[gemini] HTTP 400 with thinkingConfig; "
                      "dropping for the rest of the run")
                self.generation_config = {
                    k: v for k, v in self.generation_config.items()
                    if k != "thinkingConfig"
                }
                payload["generationConfig"] = self.generation_config
                data = self._post_with_retry(url, headers, payload)
            else:
                raise
        try:
            candidate = data["candidates"][0]
            parts = candidate.get("content", {}).get("parts", [])
        except (KeyError, IndexError) as e:
            raise ProviderError(f"gemini: unexpected response: {data}") from e
        text = "".join(p.get("text", "") for p in parts)
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise ParseError("gemini: response truncated at maxOutputTokens")
        return text


PROVIDER_CLASSES = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}


def make_provider(name: str) -> BaseProvider:
    return PROVIDER_CLASSES[name]()
