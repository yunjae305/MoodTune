"""
시맨틱 검색 로직 모듈

사용자의 자연어 감성 표현 → 임베딩 → 코사인 유사도 → Top-K 곡 반환
"""

import os
import pickle
from pathlib import Path

import numpy as np
from openai import OpenAI

from cosine import cosine_similarity_batch


CACHE_DIR = Path("cache")
EMBEDDING_MODEL = "text-embedding-3-small"
MOOD_TAG_ALIASES = {
    "새벽 감성": ["새벽", "감성적", "잔잔한"],
    "비 오는 날": ["비", "우울한", "잔잔한"],
    "그리움·이별": ["그리움", "이별", "추억", "쓸쓸한"],
    "설렘·사랑": ["설렘", "사랑", "봄", "행복한"],
    "위로·힐링": ["위로", "힐링", "따뜻한", "긍정적"],
    "신남·파티": ["신남", "파티", "에너지", "드라이브"],
    "집중·공부": ["집중", "공부", "잔잔한"],
    "여행·자유": ["여행", "자유", "드라이브"],
}


def load_cache(enriched: bool = True) -> dict:
    path = CACHE_DIR / ("enriched_embeddings.pkl" if enriched else "embeddings.pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"캐시 파일이 없습니다: {path}\n"
            "먼저 `python embed_songs.py`를 실행하세요."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    """사용자 입력 텍스트를 실시간으로 임베딩한다."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return np.array(response.data[0].embedding)


def get_mood_aliases(mood_name: str | None) -> set[str]:
    if mood_name is None:
        return set()
    if mood_name in MOOD_TAG_ALIASES:
        return set(MOOD_TAG_ALIASES[mood_name])
    return {mood_name}


def song_matches_mood(song: dict, mood_name: str | None) -> bool:
    aliases = get_mood_aliases(mood_name)
    if not aliases:
        return False
    return bool(aliases & set(song.get("mood_tags", [])))


def prioritize_indices_by_mood(
    songs: list[dict],
    similarities: np.ndarray,
    top_k: int,
    top_mood: str | None = None,
) -> list[int]:
    sorted_indices = list(np.argsort(similarities)[::-1])
    if top_mood is None:
        return sorted_indices[:top_k]

    priority_indices = [
        idx for idx in sorted_indices
        if song_matches_mood(songs[idx], top_mood)
    ]
    fallback_indices = [
        idx for idx in sorted_indices
        if not song_matches_mood(songs[idx], top_mood)
    ]
    return (priority_indices + fallback_indices)[:top_k]


def search_from_query_vector(
    query_vec: np.ndarray,
    cache: dict,
    top_k: int = 5,
    mood_filter: str | None = None,
    prioritized_mood: str | None = None,
) -> list[dict]:
    songs = cache["songs"]
    embeddings_matrix = np.array(cache["embeddings"])
    similarities = cosine_similarity_batch(query_vec, embeddings_matrix)

    if mood_filter:
        valid_indices = [
            i for i, s in enumerate(songs)
            if song_matches_mood(s, mood_filter)
        ]
        if valid_indices:
            mask = np.full(len(songs), -np.inf)
            mask[valid_indices] = similarities[valid_indices]
            similarities = mask
        prioritized_mood = None

    sorted_indices = prioritize_indices_by_mood(
        songs=songs,
        similarities=similarities,
        top_k=top_k,
        top_mood=prioritized_mood,
    )

    results = []
    for rank, idx in enumerate(sorted_indices):
        song = songs[idx].copy()
        song["similarity"] = float(similarities[idx])
        song["rank"] = rank + 1
        results.append(song)

    return results


def search(
    query: str,
    top_k: int = 5,
    enriched: bool = True,
    mood_filter: str | None = None,
    prioritized_mood: str | None = None,
) -> list[dict]:
    """
    자연어 쿼리로 시맨틱 검색을 수행한다.

    Args:
        query: 사용자 자연어 입력 (예: "비 오는 날 혼자 듣기 좋은 노래")
        top_k: 반환할 결과 수
        enriched: 풍부한 임베딩 사용 여부
        mood_filter: 특정 무드 태그로 결과 필터링 (None이면 전체)

    Returns:
        유사도 기준 상위 K개 곡 정보 딕셔너리 목록
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    cache = load_cache(enriched=enriched)

    query_vec = embed_query(client, query)
    return search_from_query_vector(
        query_vec=query_vec,
        cache=cache,
        top_k=top_k,
        mood_filter=mood_filter,
        prioritized_mood=prioritized_mood,
    )


if __name__ == "__main__":
    query = "비 오는 날 혼자 듣기 좋은 노래"
    print(f'쿼리: "{query}"\n')
    results = search(query, top_k=5)
    for r in results:
        print(f"  {r['rank']}. {r['title']} - {r['artist']} (유사도: {r['similarity']:.4f})")
        print(f"     무드: {', '.join(r['mood_tags'])}")
        print(f"     링크: {r['youtube_music_url']}\n")
