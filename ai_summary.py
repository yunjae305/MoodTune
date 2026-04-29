import os

from openai import OpenAI

from ui_reference import build_result_summary


DEFAULT_SUMMARY_MODEL = "gpt-4.1-mini"


def get_summary_model() -> str:
    return os.environ.get("OPENAI_SUMMARY_MODEL", DEFAULT_SUMMARY_MODEL).strip() or DEFAULT_SUMMARY_MODEL


def _build_summary_input(query: str, top_mood: str, top_score: float, results: list[dict]) -> str:
    top_lines = []
    for index, song in enumerate(results[:3], start=1):
        title = song.get("title", "제목 없음")
        artist = song.get("artist", "아티스트 정보 없음")
        tags = ", ".join(song.get("mood_tags", [])) or song.get("genre", "태그 없음")
        similarity = song.get("similarity")
        if similarity is None:
            score_text = "유사도 미표시"
        else:
            score_text = f"코사인 유사도 {similarity:.4f}"
        top_lines.append(f"{index}. {title} - {artist} | {tags} | {score_text}")

    return "\n".join(
        [
            f"검색 질의: {query}",
            f"가장 가까운 무드: {top_mood}",
            f"무드 유사도: {top_score:.4f}",
            "상위 추천 곡:",
            *top_lines,
        ]
    )


def generate_result_summary(
    query: str,
    top_mood: str,
    top_score: float,
    results: list[dict],
    client: OpenAI | None = None,
    model: str | None = None,
) -> str:
    top_song = results[0] if results else None
    fallback = build_result_summary(query, top_mood, top_score, top_song)
    if not results:
        return fallback

    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if model is None:
        model = get_summary_model()

    try:
        response = client.responses.create(
            model=model,
            instructions=(
                "당신은 임베딩 기반 음악 검색 결과를 설명하는 도우미입니다. "
                "한국어로 2문장 이내, 90자 안팎으로 간결하게 요약하세요. "
                "검색 질의와 가장 가까운 무드를 반영하고, 첫 번째 추천 곡을 자연스럽게 언급하세요. "
                "마크다운, 따옴표, 목록 없이 평문만 출력하세요."
            ),
            input=_build_summary_input(query, top_mood, top_score, results),
            max_output_tokens=120,
            temperature=0.4,
        )
        summary = response.output_text.strip()
        if summary:
            return summary
    except Exception:
        pass

    return fallback
