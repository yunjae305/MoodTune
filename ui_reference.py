MOOD_THEMES = {
    "새벽 감성": {
        "id": "dawn",
        "label": "새벽 감성",
        "accent": "#5c6bc0",
        "description": "잠 못 드는 새벽, 고요한 밤 공기에 어울리는 선곡",
    },
    "비 오는 날": {
        "id": "rain",
        "label": "비 오는 날",
        "accent": "#42a5f5",
        "description": "빗소리와 함께 차분하게 젖어드는 감성",
    },
    "그리움·이별": {
        "id": "longing",
        "label": "그리움·이별",
        "accent": "#ef5350",
        "description": "헤어진 뒤의 빈자리와 그리움의 온도",
    },
    "설렘·사랑": {
        "id": "love",
        "label": "설렘·사랑",
        "accent": "#ec407a",
        "description": "봄바람처럼 번지는 설렘과 달콤한 고백",
    },
    "위로·힐링": {
        "id": "healing",
        "label": "위로·힐링",
        "accent": "#66bb6a",
        "description": "지친 마음을 따스하게 감싸주는 평온함",
    },
    "신남·파티": {
        "id": "party",
        "label": "신남·파티",
        "accent": "#ffa726",
        "description": "에너지를 채워주는 경쾌하고 빠른 비트",
    },
    "집중·공부": {
        "id": "focus",
        "label": "집중·공부",
        "accent": "#26c6da",
        "description": "몰입을 도와주는 안정적이고 차분한 리듬",
    },
    "여행·자유": {
        "id": "travel",
        "label": "여행·자유",
        "accent": "#8d6e63",
        "description": "어디론가 떠나고 싶게 만드는 해방감",
    },
}

MOOD_ORDER = [
    "새벽 감성",
    "비 오는 날",
    "그리움·이별",
    "설렘·사랑",
    "위로·힐링",
    "신남·파티",
    "집중·공부",
    "여행·자유",
]

QUERY_CHIPS = [
    "드라이브 갈 때",
    "비 오는 창가에서",
    "공부할 때 듣는 노래",
    "카페 분위기",
    "운동할 때 신나는 노래",
]

NAV_ITEMS = [
    {"id": "home", "label": "검색"},
    {"id": "results", "label": "플레이리스트"},
    {"id": "map", "label": "무드 맵"},
    {"id": "compare", "label": "비교 분석"},
]


def get_mood_theme(category: str) -> dict:
    return MOOD_THEMES.get(
        category,
        {
            "id": "custom",
            "label": "커스텀",
            "accent": "#000000",
            "description": "현재 쿼리와 가장 가까운 감정 흐름",
        },
    )


def get_sidebar_moods() -> list[dict]:
    moods = []
    for category in MOOD_ORDER:
        theme = get_mood_theme(category)
        moods.append(
            {
                "category": category,
                "id": theme["id"],
                "accent": theme["accent"],
                "label": theme["label"],
            }
        )
    return moods


def truncate_text(text: str, limit: int = 72) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    clipped = normalized[: limit - 1].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped + "…"


def build_result_reason(song: dict, top_mood: str) -> str:
    tags = song.get("mood_tags", [])
    tag_text = ", ".join(tags[:2]) if tags else song.get("genre", "분위기")
    return f"{top_mood} 무드와 {tag_text} 특징이 잘 어우러집니다."


def build_result_summary(query: str, top_mood: str, top_score: float, top_song: dict | None) -> str:
    song_text = ""
    if top_song is not None:
        song_text = f" 대표 곡은 {top_song['title']} - {top_song['artist']}입니다."
    return (
        f'"{query}"는 {top_mood} 무드와 {top_score:.2f} 유사도로 추천되었습니다.'
        f"{song_text}"
    )


def build_compare_rows(simple_results: list[dict], enriched_results: list[dict]) -> list[dict]:
    row_count = max(len(simple_results), len(enriched_results))
    rows = []
    for i in range(row_count):
        rows.append(
            {
                "rank": i + 1,
                "simple": simple_results[i] if i < len(simple_results) else None,
                "enriched": enriched_results[i] if i < len(enriched_results) else None,
            }
        )
    return rows


def build_search_state_update(query: str, source: str = "input") -> dict:
    cleaned = query.strip()
    if not cleaned:
        return {}

    update = {
        "pending_query": cleaned,
        "view": "matching",
        "map_requested": False,
    }
    if source != "input":
        update["query_prefill"] = cleaned
    return update


def consume_query_prefill(state: dict) -> dict:
    query_prefill = state.get("query_prefill", "")
    if not query_prefill:
        return {}
    return {
        "query_input": query_prefill,
        "query_prefill": "",
    }
