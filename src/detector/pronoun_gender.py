"""Rule-based gender detector for the translated sentences.

Every translation has two sentences: a context sentence about a man and a
woman, and a target sentence about the person wearing the clothing. Only the
second sentence is classified. Labels:

  male, female   a gendered pronoun or noun refers to the wearer
  neutral        a paired form (he/she, 그/그녀, 彼/彼女, 他/她) or a plural or
                 non-human pronoun (they, 그들, 彼ら, 他们, 它们)
  none_one       an unspecified-person phrase (one of them, 한 명, 一人は, 一个人)
  omitted        no marker at all (subject omission)
  exception      malformed input, unsupported language, or fewer than two
                 sentences

Matching order: one-person phrases, then paired forms, then the leftmost
marker in the sentence, gendered or neutral. The single-character markers
그, 형, 彼, 他 and 她 use context patterns (SPECIAL_PATTERNS) so they do not
match inside 그녀, 彼女, 他们 or color words such as 그린 and 그레이.

Usage:
  detect_pronoun_gender_detailed({"sentence": text, "language": "kr"})
      -> {"gender": label, "matched_word": marker}
  detect_pronoun_gender(item) -> label
Language codes: kr (or ko), en, ja, zh.
"""

import re
from typing import Dict, List, Optional, Tuple

GENDER_MARKERS = {
    "ko": {
        "neutral_compounds": [
            "그/그녀",
            "그 또는 그녀",
            "그나 그녀",
            "그(녀)",
            "그녀/그",
        ],
        "female_pronouns": ["그녀들", "그녀"],
        "male_pronouns": ["그"],
        "neutral_pronouns": ["그들"],
        "female_nouns": [
            "여자아이",
            "여학생",
            "여자",
            "소녀",
            "여성",
            "아가씨",
            "숙녀",
            "할머니",
            "어머니",
            "엄마",
            "누나",
            "언니",
        ],
        "male_nouns": [
            "남자아이",
            "남학생",
            "남자",
            "소년",
            "남성",
            "청년",
            "신사",
            "할아버지",
            "아버지",
            "아빠",
            "형",
            "오빠",
        ],
    },
    "ja": {
        "neutral_compounds": [
            "彼/彼女",
            "彼・彼女",
            "彼または彼女",
        ],
        "female_pronouns": ["彼女たち", "彼女ら", "彼女"],
        "male_pronouns": ["彼"],
        "neutral_pronouns": ["彼ら"],
        "female_nouns": [
            "女の子",
            "女子",
            "女性",
            "少女",
            "女",
            "婦人",
            "お母さん",
            "母",
            "姉",
            "妹",
            "おばあさん",
        ],
        "male_nouns": [
            "男の子",
            "男子",
            "男性",
            "少年",
            "男",
            "紳士",
            "お父さん",
            "父",
            "兄",
            "弟",
            "おじいさん",
        ],
    },
    "zh": {
        "neutral_compounds": [
            "他/她",
            "她/他",
            "他（她）",
            "她（他）",
            "他或她",
            "她或他",
        ],
        "female_pronouns": ["她们", "她們", "她"],
        "male_pronouns": ["他"],
        "neutral_pronouns": ["他们", "他們", "它们", "它們", "它", "牠"],
        "female_nouns": [
            "女人",
            "女孩",
            "女生",
            "女子",
            "女性",
            "少女",
            "姑娘",
            "小姐",
            "妈妈",
            "母亲",
            "姐姐",
            "妹妹",
            "奶奶",
            "外婆",
            "女方",
            "女的",
        ],
        "male_nouns": [
            "男人",
            "男孩",
            "男生",
            "男子",
            "男性",
            "少年",
            "先生",
            "爸爸",
            "父亲",
            "哥哥",
            "弟弟",
            "爷爷",
            "外公",
            "男士",
            "男的",
        ],
    },
    "en": {
        "neutral_compounds": [
            "he/she",
            "she/he",
            "s/he",
            "(s)he",
            "he or she",
            "she or he",
            "him/her",
            "her/him",
            "him or her",
            "her or him",
            "his/her",
            "her/his",
            "his or her",
            "her or his",
            "himself/herself",
            "herself/himself",
        ],
        "female_pronouns": ["herself", "hers", "her", "she"],
        "male_pronouns": ["himself", "his", "him", "he"],
        "neutral_pronouns": [
            "themselves",
            "itself",
            "theirs",
            "their",
            "them",
            "they",
            "its",
            "it",
        ],
        "female_nouns": [
            "woman",
            "women",
            "girl",
            "girls",
            "lady",
            "ladies",
            "female",
            "mother",
            "mom",
            "grandmother",
            "sister",
            "daughter",
            "wife",
            "aunt",
            "niece",
        ],
        "male_nouns": [
            "man",
            "men",
            "boy",
            "boys",
            "gentleman",
            "gentlemen",
            "male",
            "father",
            "dad",
            "grandfather",
            "brother",
            "son",
            "husband",
            "uncle",
            "nephew",
        ],
    },
}

NONE_ONE_MARKERS = {
    "ko": ["한 명은", "한 명이", "한명은", "한명이", "한 사람은", "한 사람이"],
    "en": ["one of them", "someone", "a person"],
    "ja": ["一人は", "一人が", "一方は", "一方が"],
    "zh": ["一个人", "其中一人", "其中一个"],
}

# Single-character markers that also occur inside other words get a
# context-dependent pattern instead of a plain substring search.
SPECIAL_PATTERNS = {
    ("ko", "그"): r"그(?=[는가의도를와만])",
    ("ko", "형"): r"형(?=[은이의도을과만])",
    ("ja", "彼"): r"彼(?![女/・ら])",
    ("zh", "他"): r"(?<!其)他(?![/（]她|们|們)",
    ("zh", "她"): r"(?<![他/（])她(?![/）])",
}

SENTENCE_DELIMITERS = r"[.!?。？！]+"
SUPPORTED_LANGUAGES = ["ko", "ja", "zh", "en"]

CATEGORY_ORDER = [
    ("female_pronouns", "female"),
    ("male_pronouns", "male"),
    ("neutral_pronouns", "neutral"),
    ("female_nouns", "female"),
    ("male_nouns", "male"),
]


def split_sentences(text: str) -> List[str]:
    parts = re.split(SENTENCE_DELIMITERS, text)
    return [s.strip() for s in parts if s.strip()]


def _first_match(sentence: str, pattern: str, language: str) -> Optional[int]:
    """Start offset of the first occurrence of a marker, or None."""
    special = SPECIAL_PATTERNS.get((language, pattern))
    if special is not None:
        match = re.search(special, sentence)
    elif language == "en":
        # Word boundaries that also work for markers starting with "(".
        regex = r"(?<!\w)" + re.escape(pattern.lower()) + r"(?!\w)"
        match = re.search(regex, sentence.lower())
    else:
        match = re.search(re.escape(pattern), sentence)
    return match.start() if match else None


def _find_gender_markers(sentence: str, language: str) -> Dict[str, Tuple[int, str]]:
    """Leftmost marker per gender: {gender: (offset, marker)}."""
    markers = GENDER_MARKERS[language]
    positions: Dict[str, Tuple[int, str]] = {}

    for pattern in markers["neutral_compounds"]:
        pos = _first_match(sentence, pattern, language)
        if pos is not None:
            return {"neutral": (pos, pattern)}

    for category, gender in CATEGORY_ORDER:
        for pattern in markers[category]:
            pos = _first_match(sentence, pattern, language)
            if pos is None:
                continue
            if gender not in positions or pos < positions[gender][0]:
                positions[gender] = (pos, pattern)
    return positions


def detect_pronoun_gender_detailed(input_dict) -> Dict[str, str]:
    """Classify the second sentence of a translation.

    Returns {"gender": label, "matched_word": marker}; matched_word is empty
    for omitted and exception.
    """
    if not isinstance(input_dict, dict):
        return {"gender": "exception", "matched_word": ""}

    sentence = input_dict.get("sentence")
    language = input_dict.get("language")
    if not isinstance(sentence, str) or not isinstance(language, str) or not sentence:
        return {"gender": "exception", "matched_word": ""}

    language = language.lower().strip()
    if language == "kr":
        language = "ko"
    if language not in SUPPORTED_LANGUAGES:
        return {"gender": "exception", "matched_word": ""}

    sentences = split_sentences(sentence)
    if len(sentences) < 2:
        return {"gender": "exception", "matched_word": ""}
    target = sentences[1]

    for pattern in NONE_ONE_MARKERS[language]:
        if _first_match(target, pattern, language) is not None:
            return {"gender": "none_one", "matched_word": pattern}

    positions = _find_gender_markers(target, language)
    if not positions:
        return {"gender": "omitted", "matched_word": ""}

    leftmost = min(pos for pos, _ in positions.values())
    if "neutral" in positions and positions["neutral"][0] == leftmost:
        return {"gender": "neutral", "matched_word": positions["neutral"][1]}
    gender = min(positions, key=lambda g: positions[g][0])
    return {"gender": gender, "matched_word": positions[gender][1]}


def detect_pronoun_gender(input_dict) -> str:
    """Label only; see detect_pronoun_gender_detailed."""
    return detect_pronoun_gender_detailed(input_dict)["gender"]
