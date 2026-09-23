"""Preflight checks for the ensemble pipeline.

  python -m src.ensemble.preflight                # key check + live smoke test
  python -m src.ensemble.preflight --list-models  # list candidate model IDs

The smoke test sends one small stratified chunk per language (drawn from
units.jsonl) to every provider that has an API key and prints its labels
next to the rule-based labels.
"""

import argparse
import json

import requests

from .config import LABEL_TO_RULE, LANG_NAMES, PROVIDERS, UNITS_PATH, get_api_key
from .providers import make_provider


def pick_test_units(per_language: int = 3) -> dict[str, list[dict]]:
    if not UNITS_PATH.exists():
        raise SystemExit("units.jsonl not found - run `python -m src.ensemble.build_units` first")
    by_lang: dict[str, dict[str, dict]] = {}
    with open(UNITS_PATH, encoding="utf-8") as fh:
        for line in fh:
            unit = json.loads(line)
            slots = by_lang.setdefault(unit["language"], {})
            # prefer one unit per distinct rule label for coverage
            if unit["rule_label"] not in slots and len(slots) < per_language:
                slots[unit["rule_label"]] = unit
    return {lang: list(slots.values()) for lang, slots in by_lang.items()}


def smoke_test() -> None:
    available = [n for n in PROVIDERS if get_api_key(n)]
    missing = [n for n in PROVIDERS if not get_api_key(n)]
    for name in missing:
        envs = " or ".join(PROVIDERS[name]["env"])
        print(f"[{name}] NO KEY - set {envs} in .env")
    if not available:
        return

    tests = pick_test_units()
    for name in available:
        print(f"\n[{name}] model={PROVIDERS[name]['model']}")
        try:
            provider = make_provider(name)
            ok = mismatch = 0
            for language, units in tests.items():
                labels = provider.classify_chunk(
                    LANG_NAMES[language], [u["text"] for u in units]
                )
                for unit, label in zip(units, labels):
                    flag = "==" if LABEL_TO_RULE.get(label, label) == unit["rule_label"] else "!="
                    if flag == "==":
                        ok += 1
                    else:
                        mismatch += 1
                    print(f"  {language} rule={unit['rule_label']:<8} llm={label:<12} "
                          f"{flag} {unit['text'][:40]}")
            print(f"[{name}] OK - {ok} match, {mismatch} differ "
                  f"(differences are not necessarily errors)")
        except Exception as e:
            print(f"[{name}] FAILED: {e}")


def list_models() -> None:
    if get_api_key("anthropic"):
        import anthropic

        client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
        print("anthropic:")
        for model in client.models.list():
            print(f"  {model.id}")
    if get_api_key("openai"):
        resp = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {get_api_key('openai')}"},
            timeout=60,
        )
        resp.raise_for_status()
        print("openai (gpt-5*):")
        for model in sorted(m["id"] for m in resp.json()["data"]):
            if model.startswith("gpt-5"):
                print(f"  {model}")
    if get_api_key("gemini"):
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": get_api_key("gemini")},
            timeout=60,
        )
        resp.raise_for_status()
        print("gemini (flash*):")
        for model in resp.json().get("models", []):
            name = model["name"].removeprefix("models/")
            if "flash" in name:
                print(f"  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        list_models()
    else:
        smoke_test()


if __name__ == "__main__":
    main()
