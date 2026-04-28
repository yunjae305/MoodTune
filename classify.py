"""
제로샷 무드 분류 모듈

사용자 입력 벡터 ↔ 무드 카테고리 레이블 벡터 간 코사인 유사도로 분류
"""

import os
import pickle
from pathlib import Path

import numpy as np
from openai import OpenAI

from cosine import cosine_similarity


MOOD_LABELS_PATH = Path("cache/mood_labels.pkl")
EMBEDDING_MODEL = "text-embedding-3-small"

# 풍부한 레이블: 각 무드 카테고리를 상세히 기술하여 임베딩 품질 향상
MOOD_CATEGORIES: dict[str, str] = {
    "새벽 감성": "잠 못 자는 새벽, 혼자 있는 밤, 외로운 새벽 3시, 고요하고 쓸쓸한 감성, 밤에 혼자 듣는 노래",
    "비 오는 날": "비 오는 날 듣기 좋은, 우울하고 촉촉한 감성, 빗소리와 어울리는, 감수성 가득한 빗속 음악",
    "그리움·이별": "헤어진 연인 생각, 보고 싶다, 추억이 떠오르는, 이별 후 혼자 우는, 그리운 사람이 생각날 때",
    "설렘·사랑": "두근거리는 첫사랑, 좋아하는 사람 생각, 행복하고 달콤한, 봄날 같은 설레는 감성",
    "위로·힐링": "지쳐있을 때 듣는, 따뜻한 위로, 다 잘될 거야, 마음이 편안해지는, 힘든 하루 끝에",
    "신남·파티": "기분 좋은 드라이브, 춤추고 싶은, 에너지 넘치는, 신나고 즐거운, 파티 분위기",
    "집중·공부": "집중할 때 듣는, 가사가 방해 안 되는, 잔잔하고 반복적인 멜로디, 공부할 때 배경음악",
    "여행·자유": "훌쩍 떠나고 싶은, 탁 트인 바다나 산, 자유롭고 여유로운 감성, 여행 떠날 때",
}


def load_or_create_mood_embeddings(client: OpenAI) -> dict[str, np.ndarray]:
    """무드 카테고리 레이블 임베딩을 로드하거나 새로 생성한다."""
    if MOOD_LABELS_PATH.exists():
        with open(MOOD_LABELS_PATH, "rb") as f:
            return pickle.load(f)

    print("무드 레이블 임베딩 생성 중...")
    labels = list(MOOD_CATEGORIES.keys())
    texts = list(MOOD_CATEGORIES.values())

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    mood_embeddings = {
        label: np.array(item.embedding)
        for label, item in zip(labels, response.data)
    }

    MOOD_LABELS_PATH.parent.mkdir(exist_ok=True)
    with open(MOOD_LABELS_PATH, "wb") as f:
        pickle.dump(mood_embeddings, f)
    print(f"  무드 레이블 캐시 저장: {MOOD_LABELS_PATH}")

    return mood_embeddings


def classify_mood(query_vec: np.ndarray, mood_embeddings: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    """
    쿼리 벡터와 무드 카테고리 레이블 벡터 간 유사도를 계산하여
    유사도 순으로 정렬된 (카테고리, 유사도) 목록을 반환한다.
    """
    scores = []
    for category, label_vec in mood_embeddings.items():
        score = cosine_similarity(query_vec, label_vec)
        scores.append((category, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)


def get_top_mood(query: str, client: OpenAI | None = None) -> tuple[str, float, list[tuple[str, float]]]:
    """
    사용자 입력에 대해 가장 적합한 무드 카테고리와 전체 순위를 반환한다.

    Returns:
        (최상위 카테고리, 유사도 점수, 전체 카테고리 순위 목록)
    """
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    mood_embeddings = load_or_create_mood_embeddings(client)

    from embed_songs import load_cache  # 순환 임포트 방지
    # 쿼리 임베딩
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_vec = np.array(response.data[0].embedding)

    ranked = classify_mood(query_vec, mood_embeddings)
    top_category, top_score = ranked[0]

    return top_category, top_score, ranked


if __name__ == "__main__":
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    test_queries = [
        "비 오는 날 혼자 듣기 좋은 노래",
        "취업 준비에 지쳐서 위로받고 싶을 때",
        "새벽 드라이브에 신나게 달릴 때",
    ]

    for query in test_queries:
        top, score, ranked = get_top_mood(query, client)
        print(f'\n쿼리: "{query}"')
        print(f"  → 분류 결과: [{top}] (유사도: {score:.4f})")
        print("  전체 순위:")
        for i, (cat, s) in enumerate(ranked, 1):
            bar = "█" * int(s * 20)
            print(f"    {i}. {cat:<12} {s:.4f} {bar}")
