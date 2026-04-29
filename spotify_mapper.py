import json
import os

from openai import OpenAI


DEFAULT_SPOTIFY_MODEL = "gpt-4.1-mini"


def get_spotify_query_model() -> str:
    return os.environ.get("OPENAI_SUMMARY_MODEL", DEFAULT_SPOTIFY_MODEL).strip() or DEFAULT_SPOTIFY_MODEL


def _coerce_list(value) -> list[str]:
    if isinstance(value, list):
        items = value
    else:
        items = str(value or "").split(",")
    output = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(text)
    return output


def _clamp(value, minimum: float, maximum: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def sanitize_spotify_features(payload: dict) -> dict:
    seed_genres = _coerce_list(payload.get("seed_genres"))
    search_terms = _coerce_list(payload.get("search_terms"))

    if not seed_genres:
        seed_genres = ["pop"]
    if not search_terms:
        search_terms = seed_genres[:2] or ["pop"]

    return {
        "seed_genres": seed_genres[:3],
        "search_terms": search_terms[:3],
        "target_energy": _clamp(payload.get("target_energy"), 0.0, 1.0, 0.5),
        "target_valence": _clamp(payload.get("target_valence"), 0.0, 1.0, 0.5),
        "target_danceability": _clamp(payload.get("target_danceability"), 0.0, 1.0, 0.5),
        "target_tempo": int(_clamp(payload.get("target_tempo"), 60, 180, 120)),
    }


def keyword_override_features(query: str) -> dict:
    text = (query or "").strip().lower()
    overrides = {}

    if any(keyword in text for keyword in ["운동", "헬스", "gym", "workout"]):
        overrides = {
            "seed_genres": ["work-out", "edm", "dance"],
            "search_terms": ["workout", "gym", "energetic"],
            "target_energy": 0.9,
            "target_valence": 0.7,
            "target_danceability": 0.8,
            "target_tempo": 140,
        }
    elif any(keyword in text for keyword in ["비", "rain", "밤", "새벽", "잔잔", "calm"]):
        overrides = {
            "seed_genres": ["ambient", "chill", "indie-pop"],
            "search_terms": ["rainy night", "calm", "late night"],
            "target_energy": 0.25,
            "target_valence": 0.35,
            "target_danceability": 0.25,
            "target_tempo": 75,
        }
    elif any(keyword in text for keyword in ["드라이브", "drive"]):
        overrides = {
            "seed_genres": ["road-trip", "pop", "indie-pop"],
            "search_terms": ["night drive", "road trip", "open road"],
            "target_energy": 0.65,
            "target_valence": 0.55,
            "target_danceability": 0.55,
            "target_tempo": 115,
        }
    elif any(keyword in text for keyword in ["우울", "슬픔", "sad", "이별"]):
        overrides = {
            "seed_genres": ["sad", "ballad", "r-n-b"],
            "search_terms": ["sad", "heartbreak", "melancholy"],
            "target_energy": 0.25,
            "target_valence": 0.15,
            "target_danceability": 0.2,
            "target_tempo": 72,
        }

    if not overrides:
        return {}
    return sanitize_spotify_features(overrides)


def map_mood_to_spotify_features(query: str) -> dict:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = f"""
사용자의 감정이나 상황을 Spotify 검색에 유리한 장르와 키워드로 바꿔주세요.

입력 문장: "{query}"

아래 JSON만 반환하세요.
- seed_genres: Spotify 장르 1~3개
- search_terms: Spotify 검색용 영어 키워드 1~3개
- target_energy: 0.0 ~ 1.0
- target_valence: 0.0 ~ 1.0
- target_danceability: 0.0 ~ 1.0
- target_tempo: 60 ~ 180
"""

    try:
        response = client.chat.completions.create(
            model=get_spotify_query_model(),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        model_features = sanitize_spotify_features(json.loads(response.choices[0].message.content))
        override_features = keyword_override_features(query)
        if not override_features:
            return model_features
        merged = {
            "seed_genres": override_features["seed_genres"],
            "search_terms": override_features["search_terms"],
            "target_energy": override_features["target_energy"],
            "target_valence": override_features["target_valence"],
            "target_danceability": override_features["target_danceability"],
            "target_tempo": override_features["target_tempo"],
        }
        return sanitize_spotify_features(merged)
    except Exception as e:
        print(f"Error mapping mood: {e}")
        override_features = keyword_override_features(query)
        if override_features:
            return override_features
        return sanitize_spotify_features(
            {
                "seed_genres": ["pop"],
                "search_terms": ["pop"],
                "target_energy": 0.5,
                "target_valence": 0.5,
                "target_danceability": 0.5,
                "target_tempo": 120,
            }
        )
