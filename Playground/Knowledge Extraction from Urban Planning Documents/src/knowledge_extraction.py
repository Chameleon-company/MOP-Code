import re

TARGET_TERMS = [
    "planning scheme",
    "zone",
    "overlay",
    "planning permit",
    "responsible authority",
    "planning authority",
    "schedule"
]

NOISE_PATTERNS = [
    r"contents",
    r"page\s*\|",
    r"\.{3,}",
]

OTHER_TERMS = [
    "planning scheme",
    "zone",
    "overlay",
    "planning permit",
    "responsible authority",
    "planning authority",
    "schedule"
]


def split_into_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def is_noise(sentence: str) -> bool:
    s = sentence.lower().strip()

    for pattern in NOISE_PATTERNS:
        if re.search(pattern, s):
            return True

    if len(s) < 6:
        return True

    return False


def clean_candidate(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r"^[\d\.\-\s]+", "", sentence)
    return sentence


def is_heading_like(sentence: str, term: str) -> bool:
    s = sentence.lower().strip()
    return (
        s == f"what is a {term}?"
        or s == f"what is an {term}?"
        or s.startswith(f"what is a {term}?")
        or s.startswith(f"what is an {term}?")
    )


def contains_too_many_other_terms(text: str, current_term: str) -> bool:
    text_lower = text.lower()
    count = 0
    for term in OTHER_TERMS:
        if term != current_term and term in text_lower:
            count += 1
    return count >= 2


def score_sentence(sentence: str, term: str) -> int:
    s = sentence.lower()
    score = 0

    if term in s:
        score += 3

    if "what is" in s:
        score += 2

    if "is a" in s or "is an" in s or "means" in s:
        score += 2

    if "control" in s or "identify" in s or "purpose" in s or "additional" in s:
        score += 1

    if len(sentence) < 300:
        score += 2
    elif len(sentence) < 600:
        score += 1

    if is_noise(sentence):
        score -= 3

    return score


def build_context(sentences: list, index: int, term: str) -> str:
    current = clean_candidate(sentences[index])

    if is_heading_like(current, term):
        collected = [current]

        for j in range(index + 1, min(len(sentences), index + 4)):
            next_sent = clean_candidate(sentences[j])

            if is_noise(next_sent):
                continue

            if next_sent.lower().startswith("what is "):
                break

            if contains_too_many_other_terms(next_sent, term):
                break

            collected.append(next_sent)

            if len(" ".join(collected)) > 450:
                break

        return " ".join(collected)

    start = max(0, index - 1)
    end = min(len(sentences), index + 2)
    context = " ".join(clean_candidate(sentences[j]) for j in range(start, end))
    return context


def find_best_definition(sentences: list, term: str) -> str:
    candidates = []

    for i, sentence in enumerate(sentences):
        s = clean_candidate(sentence)

        if term.lower() in s.lower() and not is_noise(s):
            context = build_context(sentences, i, term)
            candidates.append((score_sentence(context, term), context))

    if not candidates:
        return "Not specified"

    candidates.sort(key=lambda x: (-x[0], len(x[1])))
    return candidates[0][1]


def infer_purpose(text: str, term: str) -> str:
    t = text.lower()

    if term == "planning scheme":
        if "control" in t or "land use" in t or "development" in t:
            return "Controls land use and development."
    elif term == "zone":
        if "identify" in t or "use" in t or "land" in t:
            return "Designates land for particular uses."
    elif term == "overlay":
        if "additional" in t or "specific" in t or "applies" in t:
            return "Applies additional controls for specific issues."
    elif term == "planning permit":
        if "permit" in t or "approval" in t or "grant" in t:
            return "Provides approval for a use or development."
    elif term == "responsible authority":
        if "decide" in t or "application" in t or "administer" in t:
            return "Assesses or decides planning applications."
    elif term == "planning authority":
        if "prepare" in t or "amend" in t or "scheme" in t:
            return "Prepares or amends the planning scheme."
    elif term == "schedule":
        if "local" in t or "objectives" in t or "requirements" in t:
            return "Adds detailed or local provisions to controls."

    return "Not specified"


def infer_permit_info(text: str) -> str:
    t = text.lower()
    if "planning permit" in t:
        return "Planning permit mentioned."
    if "subject to a permit" in t or "subject to planning permit" in t:
        return "Permit required or conditionally required."
    if "without planning approval" in t:
        return "May not require planning approval in some cases."
    if "permit is required" in t:
        return "Permit requirement identified."
    return "Not specified"


def infer_authority(text: str) -> str:
    t = text.lower()
    found = []

    if "responsible authority" in t:
        found.append("responsible authority")
    if "planning authority" in t:
        found.append("planning authority")
    if "council" in t:
        found.append("council")
    if "minister" in t:
        found.append("minister")
    if "vcat" in t:
        found.append("vcat")

    if found:
        return ", ".join(sorted(set(found)))

    return "Not specified"


def infer_related_components(text: str) -> str:
    t = text.lower()
    comps = []

    for item in [
        "zone",
        "overlay",
        "schedule",
        "planning scheme",
        "planning permit",
        "vpp",
        "council"
    ]:
        if item in t:
            comps.append(item)

    if comps:
        return ", ".join(sorted(set(comps)))

    return "Not specified"


def infer_source_section(text: str) -> str:
    match = re.search(r'\b(\d+\.\d+(?:\.\d+)?)\b', text)
    if match:
        return match.group(1)
    return "Not specified"


def extract_knowledge(text: str, document_name: str) -> list:
    sentences = split_into_sentences(text)
    results = []

    for term in TARGET_TERMS:
        definition = find_best_definition(sentences, term)

        record = {
            "document_name": document_name,
            "term": term,
            "definition": definition,
            "purpose_or_function": infer_purpose(definition, term),
            "permit_related_information": infer_permit_info(definition),
            "authority_involved": infer_authority(definition),
            "related_components": infer_related_components(definition),
            "source_section": infer_source_section(definition)
        }

        results.append(record)

    return results