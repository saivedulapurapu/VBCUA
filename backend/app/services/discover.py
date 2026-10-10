"""
Concept discovery: "Trending today" topics (Gemini + Google Search, cached once per day) and
AI-written reference definitions for any topic a user wants to explain.
"""
import logging
import random
import re
import threading
import time
from datetime import date as Date
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.models.concept import ReferenceConcept, TrendingTopics
from backend.app.services import llm_client

logger = logging.getLogger(__name__)

CATEGORIES = ("News", "Science", "Technology", "Business", "Sports", "Culture", "Health", "Environment")
TOPIC_COUNT = 8
# After a failed grounded search, wait before trying again (the curated list is shown meanwhile)
RETRY_AFTER_FAILURE_S = 30 * 60

# Shown when trending topics are unavailable (no key, quota, outage): a new random set on every
# request. Never labelled "trending".
CURATED_TOPICS: List[Dict[str, str]] = [
    {"title": "How elections work in a democracy", "category": "News"},
    {"title": "Artificial intelligence regulation", "category": "News"},
    {"title": "The role of the United Nations", "category": "News"},
    {"title": "Freedom of the press", "category": "News"},
    {"title": "Refugees and migration", "category": "News"},
    {"title": "How a budget is passed by parliament", "category": "News"},
    {"title": "Space missions to the Moon", "category": "News"},
    {"title": "Cyber attacks on governments", "category": "News"},
    {"title": "The James Webb Space Telescope", "category": "Science"},
    {"title": "Plate tectonics and earthquakes", "category": "Science"},
    {"title": "Quantum computing", "category": "Science"},
    {"title": "CRISPR gene editing", "category": "Science"},
    {"title": "Black holes", "category": "Science"},
    {"title": "How the human brain stores memories", "category": "Science"},
    {"title": "The origin of the universe", "category": "Science"},
    {"title": "Volcanoes and eruptions", "category": "Science"},
    {"title": "How large language models work", "category": "Technology"},
    {"title": "Cryptocurrency and blockchain", "category": "Technology"},
    {"title": "Cybersecurity and phishing", "category": "Technology"},
    {"title": "Electric vehicles and batteries", "category": "Technology"},
    {"title": "5G mobile networks", "category": "Technology"},
    {"title": "UPI and digital payments", "category": "Technology"},
    {"title": "Self-driving cars", "category": "Technology"},
    {"title": "How search engines rank pages", "category": "Technology"},
    {"title": "Inflation and interest rates", "category": "Business"},
    {"title": "Supply chains and global trade", "category": "Business"},
    {"title": "The stock market", "category": "Business"},
    {"title": "Startups and venture capital", "category": "Business"},
    {"title": "How e-commerce delivers in a day", "category": "Business"},
    {"title": "Gig economy and freelancing", "category": "Business"},
    {"title": "Why oil prices change", "category": "Business"},
    {"title": "Brand marketing on social media", "category": "Business"},
    {"title": "The Olympic Games", "category": "Sports"},
    {"title": "Cricket World Cup", "category": "Sports"},
    {"title": "T20 cricket and the IPL", "category": "Sports"},
    {"title": "Football World Cup", "category": "Sports"},
    {"title": "Sports science and athlete training", "category": "Sports"},
    {"title": "Doping in sport", "category": "Sports"},
    {"title": "Chess and computer engines", "category": "Sports"},
    {"title": "The economics of sports leagues", "category": "Sports"},
    {"title": "Streaming and the film industry", "category": "Culture"},
    {"title": "Social media and misinformation", "category": "Culture"},
    {"title": "The rise of K-pop", "category": "Culture"},
    {"title": "Indian classical dance", "category": "Culture"},
    {"title": "Video games as an art form", "category": "Culture"},
    {"title": "Museums in the digital age", "category": "Culture"},
    {"title": "Festivals and their history", "category": "Culture"},
    {"title": "How memes spread online", "category": "Culture"},
    {"title": "How vaccines train the immune system", "category": "Health"},
    {"title": "Antibiotic resistance", "category": "Health"},
    {"title": "Mental health and sleep", "category": "Health"},
    {"title": "Air pollution and health", "category": "Health"},
    {"title": "Diabetes and lifestyle", "category": "Health"},
    {"title": "How pandemics spread", "category": "Health"},
    {"title": "Nutrition and balanced diets", "category": "Health"},
    {"title": "Screen time and eyesight", "category": "Health"},
    {"title": "Climate change and global warming", "category": "Environment"},
    {"title": "Renewable energy: solar and wind power", "category": "Environment"},
    {"title": "Water scarcity", "category": "Environment"},
    {"title": "Plastic pollution in the oceans", "category": "Environment"},
    {"title": "Deforestation and biodiversity", "category": "Environment"},
    {"title": "Electric grids and energy storage", "category": "Environment"},
    {"title": "Heatwaves and extreme weather", "category": "Environment"},
    {"title": "Recycling and the circular economy", "category": "Environment"},
]

_fetch_lock = threading.Lock()
_last_failure: Dict[str, float] = {}


def today() -> str:
    return Date.today().isoformat()


def curated_topics(count: int = TOPIC_COUNT, rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    """A fresh random selection on every request, spread across categories where possible."""
    rng = rng or random.Random()
    by_category: Dict[str, List[Dict[str, str]]] = {}
    for topic in CURATED_TOPICS:
        by_category.setdefault(topic["category"], []).append(topic)
    categories = list(by_category)
    rng.shuffle(categories)
    picks = [rng.choice(by_category[c]) for c in categories[:count]]
    remaining = [t for t in CURATED_TOPICS if t not in picks]
    picks += rng.sample(remaining, k=max(0, min(count - len(picks), len(remaining))))
    return [{"title": t["title"], "category": t["category"], "why": ""} for t in picks]


def trending_prompt(day: str) -> str:
    return f"""Today is {day}. Use Google Search to find {TOPIC_COUNT} topics that are trending in the news today, across
different areas (for example world news, science, technology, business, sports, culture, health, environment).
Choose topics a student could explain out loud in one or two minutes, and avoid graphic or purely celebrity-gossip items.
Return ONLY a JSON array, no other text, in this form:
[{{"title": "short topic title (max 8 words)", "category": "one of {', '.join(CATEGORIES)}", "why": "one sentence on why it is in the news today"}}]"""


def clean_topics(data: Any) -> List[Dict[str, str]]:
    """Validates the model's topic list; drops malformed entries and duplicates."""
    if isinstance(data, dict):
        data = data.get("topics", [])
    if not isinstance(data, list):
        raise llm_client.LLMError("Trending topics answer is not a list")
    topics, seen = [], set()
    for item in data:
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title", "")).split())[:90]
        if not title or title.lower() in seen:
            continue
        category = str(item.get("category", "")).strip().title()
        topics.append({
            "title": title,
            "category": category if category in CATEGORIES else "News",
            "why": " ".join(str(item.get("why", "")).split())[:240],
        })
        seen.add(title.lower())
        if len(topics) == TOPIC_COUNT:
            break
    if len(topics) < 3:
        raise llm_client.LLMError("Too few valid trending topics")
    return topics


def _row_to_response(row: TrendingTopics) -> Dict[str, Any]:
    return {
        "date": row.date,
        "origin": row.origin,
        "topics": row.topics,
        "sources": row.sources or [],
        "search_entry_point": row.search_entry_point,
    }


def _curated_response(day: str) -> Dict[str, Any]:
    return {"date": day, "origin": "curated", "topics": curated_topics(), "sources": [], "search_entry_point": None}


def get_trending(db: Session, day: Optional[str] = None) -> Dict[str, Any]:
    """
    Today's topics: cached grounded-search results if available, else one grounded search
    (at most one at a time, and not retried for 30 minutes after a failure), else the curated list.
    """
    day = day or today()
    row = db.get(TrendingTopics, day)
    if row:
        return _row_to_response(row)
    if not llm_client.is_configured():
        return _curated_response(day)
    with _fetch_lock:
        row = db.get(TrendingTopics, day)  # another request may have fetched it meanwhile
        if row:
            return _row_to_response(row)
        if time.time() - _last_failure.get(day, 0) < RETRY_AFTER_FAILURE_S:
            return _curated_response(day)
        try:
            start = time.perf_counter()
            data, grounding = llm_client.generate_grounded_json(trending_prompt(day), model=settings.GEMINI_DISCOVER_MODEL)
            topics = clean_topics(data)
            if not grounding["sources"]:
                raise llm_client.LLMError("Answer was not grounded in search results")
            row = TrendingTopics(
                date=day, origin="google-search", topics=topics, sources=grounding["sources"][:12],
                search_entry_point=grounding["search_entry_point"], model=settings.GEMINI_DISCOVER_MODEL,
            )
            db.add(row)
            db.commit()
            logger.info(f"Trending topics for {day} from {settings.GEMINI_DISCOVER_MODEL} in {time.perf_counter() - start:.1f}s")
            return _row_to_response(row)
        except Exception as e:
            db.rollback()
            _last_failure[day] = time.time()
            logger.warning(f"Trending topics unavailable, showing curated topics: {e}")
            return _curated_response(day)


def normalise_topic(topic: str) -> str:
    topic = " ".join(str(topic).split())
    if not 2 <= len(topic) <= 120:
        raise ValueError("Enter a topic between 2 and 120 characters.")
    return topic


def reference_prompt(topic: str, day: str) -> str:
    return f"""Today is {day}. A student will explain the topic below out loud for one or two minutes. Write the reference
that their explanation will be compared with. If the topic is about current events, use Google Search for up-to-date facts.

Topic: {topic}

Return ONLY a JSON object, no other text:
{{"title": "the topic's correct, properly spelled name (fix any spelling mistakes in the topic above; max 8 words)",
  "description": "a factual, neutral reference explanation of 80-150 words covering the key ideas a good explanation should include",
  "keywords": ["5 to 8 short key terms (1-3 words each) that a good explanation should mention"]}}"""


def clean_reference(data: Any, fallback_title: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise llm_client.LLMError("Reference answer is not an object")
    title = " ".join(str(data.get("title") or fallback_title).split())[:100]
    description = " ".join(str(data.get("description", "")).split())
    raw_keywords = data.get("keywords") if isinstance(data.get("keywords"), list) else []
    keywords, seen = [], set()
    for kw in raw_keywords:
        kw = re.sub(r"\s+", " ", str(kw)).strip().strip(".,;")[:40]
        if kw and kw.lower() not in seen:
            keywords.append(kw)
            seen.add(kw.lower())
    if len(description) < 40 or len(keywords) < 3:
        raise llm_client.LLMError("Reference answer is incomplete")
    return {"title": title, "description": description[:1500], "keywords": keywords[:10]}


def find_concept(db: Session, title: str) -> Optional[ReferenceConcept]:
    return db.query(ReferenceConcept).filter(func.lower(ReferenceConcept.title) == title.lower()).first()


def generate_concept(db: Session, topic: str) -> ReferenceConcept:
    """
    Returns an existing concept with this title, or creates one whose reference definition and
    keywords are written by Gemini (source "ai-generated"). Raises ValueError for a bad topic,
    llm_client.LLMNotConfigured without a key and llm_client.LLMError if generation fails.
    """
    topic = normalise_topic(topic)
    existing = find_concept(db, topic)
    if existing:
        return existing
    prompt = reference_prompt(topic, today())
    try:
        data, _grounding = llm_client.generate_grounded_json(prompt, model=settings.GEMINI_DISCOVER_MODEL)
    except llm_client.LLMNotConfigured:
        raise
    except llm_client.LLMError as e:
        # Grounding unavailable (e.g. no search quota on this key): write the reference from the model's own knowledge
        logger.warning(f"Grounded reference failed ({e}); writing it without search")
        data = llm_client.generate_json(prompt.replace("If the topic is about current events, use Google Search for up-to-date facts.", ""),
                                        model=settings.GEMINI_DISCOVER_MODEL)
    reference = clean_reference(data, topic)
    existing = find_concept(db, reference["title"])
    if existing:
        return existing
    concept = ReferenceConcept(
        title=reference["title"], description=reference["description"], keywords=reference["keywords"], source="ai-generated",
    )
    db.add(concept)
    db.commit()
    db.refresh(concept)
    logger.info(f"AI-generated reference for {concept.title!r} with {len(concept.keywords)} keywords")
    return concept
