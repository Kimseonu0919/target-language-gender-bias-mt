"""Google Cloud Translation API (Basic, v2) over REST with an API key."""

import html

import requests

API_URL = "https://translation.googleapis.com/language/translate/v2"
CHUNK_SIZE = 100  # the API accepts at most 128 segments per request


class GoogleTrans:
    def __init__(self, api_key):
        self.api_key = api_key

    def translate_batch(self, sentences, src_lang, target_lang):
        """Translate a list of sentences; returns [] if any chunk fails."""
        if target_lang == "zh":
            target_lang = "zh-CN"

        translated = []
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk = sentences[i : i + CHUNK_SIZE]
            params = {
                "q": chunk,
                "source": src_lang,
                "target": target_lang,
                "format": "text",
                "key": self.api_key,
            }
            response = None
            try:
                response = requests.post(API_URL, data=params, timeout=60)
                response.raise_for_status()
                items = response.json()["data"]["translations"]
                translated.extend(html.unescape(t["translatedText"]) for t in items)
            except Exception as e:
                print(f"[google] chunk {i} ({src_lang} -> {target_lang}) failed: {e}")
                if response is not None:
                    print(response.text[:500])
                return []
        return translated
