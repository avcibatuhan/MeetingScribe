"""
Words Whisper tends to get wrong - names, products, jargon, acronyms.

These are fed to Whisper as a hint before each chunk, which makes it far
more likely to spell them correctly instead of guessing something
phonetically similar ("Kubernetes" -> "cooper netties").

Just edit the lists below. Keep each entry short and spelled exactly the
way you want it to appear in the transcript.

Tips:
  - Add names of people who join your meetings often.
  - Add product, company and project names.
  - Add acronyms your team says out loud.
  - Don't add ordinary words - it only helps for unusual ones.
  - Keep the total under ~200 terms; a huge list dilutes the effect.
"""

PEOPLE = [
    "Winchester",
    # "Ayse Yilmaz",
    # "Mehmet Demir",
]

COMPANIES = [
    "Anthropic",
    "Deepgram",
    "HuggingFace",
    # "Acme Corp",
]

PRODUCTS = [
    "Whisper",
    "Gemini",
    "Claude",
    "PySide",
    "PipeWire",
    # "Jira",
    # "Confluence",
]

TECHNICAL = [
    "Kubernetes",
    "Postgres",
    "Redis",
    "API",
    "CI/CD",
    "webhook",
    "endpoint",
    "latency",
    "throughput",
    "quantization",
    "diarization",
]

BUSINESS = [
    "Q1", "Q2", "Q3", "Q4",
    "KPI",
    "OKR",
    "churn",
    "ARR",
    "MRR",
    "runway",
    "onboarding",
    "roadmap",
    "stakeholder",
    "retrospective",
    "standup",
    "sprint",
]

# Anything that does not fit above.
OTHER = [
]


def all_terms():
    """Every term, de-duplicated, order preserved."""
    seen, out = set(), []
    for group in (PEOPLE, COMPANIES, PRODUCTS, TECHNICAL, BUSINESS, OTHER):
        for t in group:
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
    return out


def prompt(limit=180):
    """The hint string handed to Whisper as initial_prompt."""
    terms = all_terms()[:limit]
    if not terms:
        return ""
    return "Terms used in this meeting: " + ", ".join(terms) + "."


if __name__ == "__main__":
    print(f"{len(all_terms())} terms\n")
    print(prompt())
