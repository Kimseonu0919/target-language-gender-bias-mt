"""Run the LLM ensemble over all unique units, with checkpointing.

Usage:
  python -m src.ensemble.run_ensemble                       # all providers with keys
  python -m src.ensemble.run_ensemble --providers anthropic,gemini
  python -m src.ensemble.run_ensemble --limit 40            # smoke test
  python -m src.ensemble.run_ensemble --dry-run             # no API calls

Results are appended to results/validation/raw/<provider>.jsonl, one line
per unit: {unit_id, label, model, ts}. Units already present in that file
are skipped, so the script is safe to interrupt and resume. The repo ships
the paper run's files there; to redo the run, move them aside first,
otherwise every provider reports 0 units to do.

--dry-run prints, per provider, the model ID, whether an API key is set,
the units still to classify out of the selected units, and the number of
requests that would be sent (one request per chunk of --chunk-size units
of one language). Nothing is sent.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .config import (
    CHUNK_SIZE,
    CONCURRENCY,
    LANG_NAMES,
    PROVIDERS,
    RAW_DIR,
    UNITS_PATH,
    get_api_key,
)
from .prompts import ParseError
from .providers import ProviderError, make_provider


def load_units() -> list[dict]:
    if not UNITS_PATH.exists():
        raise SystemExit("units.jsonl not found - run `python -m src.ensemble.build_units` first")
    with open(UNITS_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def load_done(provider: str) -> set[str]:
    path = RAW_DIR / f"{provider}.jsonl"
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["unit_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def classify_chunk_or_split(provider, language_name, chunk):
    """Classify a chunk; after two failed attempts, split and recurse.

    Returns (results, failures): lists of (unit, label) and units.
    """
    for _ in range(2):
        try:
            labels = provider.classify_chunk(language_name, [u["text"] for u in chunk])
            return list(zip(chunk, labels)), []
        except (ParseError, ProviderError):
            continue
    if len(chunk) == 1:
        return [], list(chunk)
    mid = len(chunk) // 2
    left_ok, left_bad = classify_chunk_or_split(provider, language_name, chunk[:mid])
    right_ok, right_bad = classify_chunk_or_split(provider, language_name, chunk[mid:])
    return left_ok + right_ok, left_bad + right_bad


def make_chunks(units: list[dict], chunk_size: int) -> list[tuple[str, list[dict]]]:
    """Group units by language and cut each group into request-sized chunks."""
    by_lang: dict[str, list[dict]] = {}
    for unit in units:
        by_lang.setdefault(unit["language"], []).append(unit)
    chunks = []
    for language, lang_units in by_lang.items():
        for i in range(0, len(lang_units), chunk_size):
            chunks.append((LANG_NAMES[language], lang_units[i : i + chunk_size]))
    return chunks


def run_provider(name: str, units: list[dict], chunk_size: int, concurrency: int) -> None:
    provider = make_provider(name)
    done = load_done(name)
    todo = [u for u in units if u["unit_id"] not in done]
    print(f"[{name}] model={provider.model} done={len(done)} todo={len(todo)}")
    if not todo:
        return

    chunks = make_chunks(todo, chunk_size)
    out_path = RAW_DIR / f"{name}.jsonl"
    lock = threading.Lock()
    stats = {"units": 0, "failed": 0, "chunks": 0}

    def work(job):
        language_name, chunk = job
        results, failures = classify_chunk_or_split(provider, language_name, chunk)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with lock:
            with open(out_path, "a", encoding="utf-8", newline="\n") as fh:
                for unit, label in results:
                    fh.write(
                        json.dumps(
                            {
                                "unit_id": unit["unit_id"],
                                "label": label,
                                "model": provider.model,
                                "ts": timestamp,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            stats["units"] += len(results)
            stats["failed"] += len(failures)
            stats["chunks"] += 1
            if stats["chunks"] % 25 == 0 or stats["chunks"] == len(chunks):
                print(
                    f"[{name}] chunks {stats['chunks']}/{len(chunks)} "
                    f"units {stats['units']} failed {stats['failed']}"
                )
        return failures

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, job) for job in chunks]
        failed_units = [u for f in as_completed(futures) for u in f.result()]

    if failed_units:
        print(f"[{name}] WARNING: {len(failed_units)} units failed after retries; "
              f"re-run to retry them")
    else:
        print(f"[{name}] complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--providers", default=None,
                        help="comma-separated subset (default: all with API keys)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None,
                        help="classify only the first N units (smoke test)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    units = load_units()
    if args.limit:
        units = units[: args.limit]

    if args.providers:
        names = [n.strip() for n in args.providers.split(",")]
        unknown = [n for n in names if n not in PROVIDERS]
        if unknown:
            raise SystemExit(f"unknown providers: {unknown}")
    elif args.dry_run:
        names = list(PROVIDERS)
    else:
        names = [n for n in PROVIDERS if get_api_key(n)]
        missing = [n for n in PROVIDERS if not get_api_key(n)]
        if missing:
            print(f"skipping providers without API keys: {', '.join(missing)}")
    if not names:
        raise SystemExit("no providers available - add API keys to .env")

    if args.dry_run:
        for name in names:
            done = load_done(name)
            todo = [u for u in units if u["unit_id"] not in done]
            print(f"{name}: model={PROVIDERS[name]['model']} "
                  f"key={'yes' if get_api_key(name) else 'no'} "
                  f"todo={len(todo)}/{len(units)} units, "
                  f"{len(make_chunks(todo, args.chunk_size))} requests")
        return

    for name in names:
        run_provider(name, units, args.chunk_size, args.concurrency)


if __name__ == "__main__":
    main()
