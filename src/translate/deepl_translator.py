"""DeepL API client through the official deepl package."""

CHUNK_SIZE = 50  # keeps each request under the size limit (HTTP 413)


class DeeplTrans:
    def __init__(self, api_key):
        import deepl  # imported here so the module loads without the package

        self.translator = deepl.Translator(api_key)

    def translate_batch(self, sentences, src_lang, target_lang):
        """Translate a list of sentences; returns [] if any chunk fails."""
        target_lang = target_lang.upper()
        if target_lang == "EN":
            target_lang = "EN-US"  # DeepL requires a regional variant for English

        translated = []
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk = sentences[i : i + CHUNK_SIZE]
            try:
                results = self.translator.translate_text(
                    chunk, source_lang=src_lang.upper(), target_lang=target_lang
                )
            except Exception as e:
                print(f"[deepl] chunk {i} ({src_lang} -> {target_lang}) failed: {e}")
                return []
            translated.extend(r.text for r in results)
        return translated
