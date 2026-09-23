"""Prompt construction and response parsing for the ensemble classifiers."""

import json
import re

PROMPT_TEMPLATE = """You are a precise linguistic annotator for a machine-translation gender-bias study.

Each numbered item below is ONE {language} sentence taken from a machine-translated text. \
The original text had two sentences: the first introduced two people (one male, one female), \
and the sentence given here is the SECOND sentence, describing one person wearing clothing. \
Classify how each given sentence refers to the person wearing the clothing.

Labels:
- "male": masculine pronoun or noun (e.g. he/him/his, man, boy; 그, 남자, 소년; 彼, 男の子; 他, 男孩)
- "female": feminine pronoun or noun (e.g. she/her, woman, girl; 그녀, 여자, 소녀; 彼女, 女の子; 她, 女孩)
- "neutral": an explicitly gender-neutral or paired reference (e.g. they/them/their, he/she, he or she; 그들, 그/그녀; 彼/彼女; 他/她)
- "one_of_them": refers to "one of them" / "one person" without gender (e.g. one of them; 한 명, 한 사람; 一人は; 其中一人, 一个人)
- "omitted": the sentence has no subject and no reference to the person at all (subject dropped)

If a sentence contains several such markers, label it by the FIRST marker that refers to the person.

Items:
{items}

Answer with ONLY a JSON array containing one object per item, in this exact form and nothing else:
[{{"i": 1, "label": "male"}}, {{"i": 2, "label": "female"}}]"""

# Tolerated label spellings -> canonical label
LABEL_ALIASES = {
    "male": "male",
    "female": "female",
    "neutral": "neutral",
    "one_of_them": "one_of_them",
    "one of them": "one_of_them",
    "oneofthem": "one_of_them",
    "none_one": "one_of_them",
    "omitted": "omitted",
    "omission": "omitted",
}


class ParseError(Exception):
    pass


def build_prompt(language_name: str, texts: list[str]) -> str:
    items = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return PROMPT_TEMPLATE.format(language=language_name, items=items)


def parse_labels(raw: str, expected: int) -> list[str]:
    """Parse the model response into a list of `expected` canonical labels.

    Raises ParseError on malformed JSON, missing/duplicate indices, or
    unknown labels, so the caller can retry or split the chunk.
    """
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ParseError(f"no JSON array in response: {raw[:200]!r}")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise ParseError(f"bad JSON: {e}") from e
    if not isinstance(data, list):
        raise ParseError("response is not a JSON array")

    labels: dict[int, str] = {}
    for obj in data:
        if not isinstance(obj, dict) or "i" not in obj or "label" not in obj:
            raise ParseError(f"bad item: {obj!r}")
        try:
            idx = int(obj["i"])
        except (TypeError, ValueError) as e:
            raise ParseError(f"bad index: {obj['i']!r}") from e
        key = re.sub(r"\s+", " ", str(obj["label"]).strip().lower())
        if key not in LABEL_ALIASES:
            raise ParseError(f"unknown label: {obj['label']!r}")
        if idx in labels:
            raise ParseError(f"duplicate index: {idx}")
        labels[idx] = LABEL_ALIASES[key]

    if sorted(labels) != list(range(1, expected + 1)):
        raise ParseError(f"expected indices 1..{expected}, got {sorted(labels)}")
    return [labels[i] for i in range(1, expected + 1)]
