import re 

REASONING_KEYWORDS = {
    "explain",
    "analyze",
    "compare",
    "evaluate",
    "design",
    "reason",
    "justify",
    "debug",
    "optimize",
    "trade-off",
    "tradeoff",
    "architecture",
}


TECHNICAL_KEYWORDS = {
    "api",
    "database",
    "redis",
    "docker",
    "kubernetes",
    "async",
    "fastapi",
    "python",
    "algorithm",
    "distributed",
    "latency",
    "scalability",
    "race condition",
    "vector database",
    "rag",
    "llm",
}


def extract_features(prompt: str) -> dict:

    prompt_lower = prompt.lower()

    words = re.findall(
        r"\b\w+\b",
        prompt_lower
    )

    sentences = re.split(
        r"[.!?]+",
        prompt
    )

    sentences = [
        sentence
        for sentence in sentences
        if sentence.strip()
    ]

    reasoning_count = sum(
        1
        for keyword in REASONING_KEYWORDS
        if keyword in prompt_lower
    )

    technical_count = sum(
        1
        for keyword in TECHNICAL_KEYWORDS
        if keyword in prompt_lower
    )

    has_code = int(
        "```" in prompt
        or "def " in prompt
        or "class " in prompt
        or "import " in prompt
    )

    return {
        "word_count": len(words),

        "character_count": len(prompt),

        "sentence_count": len(sentences),

        "question_count": prompt.count("?"),

        "newline_count": prompt.count("\n"),

        "reasoning_keyword_count":
            reasoning_count,

        "technical_keyword_count":
            technical_count,

        "has_code": has_code,
    }