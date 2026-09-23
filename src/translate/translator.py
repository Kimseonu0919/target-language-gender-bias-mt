"""Translate a generated sentence file with DeepL and Google Translate.

Usage: python -m src.translate.translator <input.json> [--out DIR]

The input is a data/generated/{default|color}_{src}.json file from the
preprocessing step. Each system translates it into every language in
TARGET_LANGS, one thread per (system, language), and writes
{system}_{type}_{src}_{target}.json to data/generated/translations/ (or
--out). Keys are read from .env in the repository root (see .env.example);
a system without a key is skipped, and the process exits with status 1 if
any task fails.
"""

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .deepl_translator import DeeplTrans
from .google_translator import GoogleTrans

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "generated" / "translations"

# File-name code -> API code per system. "kr" is the code the rest of the
# repository uses for Korean.
TARGET_LANGS = {
    "kr": {"google": "ko", "deepl": "KO"},
    "ja": {"google": "ja", "deepl": "JA"},
    "zh": {"google": "zh-CN", "deepl": "ZH"},
    "en": {"google": "en", "deepl": "EN-US"},
}


def load_api_keys():
    load_dotenv(ROOT / ".env")
    deepl_key = os.getenv("DEEPL_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")  # Cloud Translation API key string
    return deepl_key, google_key


def translate(system, client, sentences, src, target, api_target, prefix, out_dir):
    """Runs in a worker thread; raises if the system returned nothing."""
    print(f"[{system}] translating to {target}")
    texts = client.translate_batch(sentences, src, api_target)
    if not texts:
        raise RuntimeError(f"{system}: no result for {target}")
    rows = [{"sentence": text, "language": target} for text in texts]
    path = out_dir / f"{system}_{prefix}_{target}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=4)
    print(f"[{system}] {len(rows)} sentences -> {path.name}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="{default|color}_{src}.json from preprocessing")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"input file not found: {input_path}")
    prefix = input_path.stem  # e.g. default_tr
    if "_" not in prefix:
        sys.exit(f"expected a {{type}}_{{src}}.json file, got {input_path.name}")
    src = prefix.rsplit("_", 1)[1]

    deepl_key, google_key = load_api_keys()
    clients = {}
    if deepl_key:
        clients["deepl"] = DeeplTrans(deepl_key)
    if google_key:
        clients["google"] = GoogleTrans(google_key)
    if not clients:
        sys.exit("no API keys found (DEEPL_API_KEY, GOOGLE_API_KEY); see .env.example")

    with open(input_path, encoding="utf-8") as fh:
        sentences = [item["sentence"] for item in json.load(fh)]
    print(f"{len(sentences)} sentences from {input_path.name}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(translate, system, client, sentences, src, target,
                            codes[system], prefix, out_dir)
            for target, codes in TARGET_LANGS.items()
            for system, client in clients.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                failed += 1
                print(f"failed: {e}")

    if failed:
        sys.exit(f"{failed} of {len(futures)} tasks failed")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
