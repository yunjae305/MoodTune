import os
import pickle
from pathlib import Path

import numpy as np
from openai import OpenAI

from cosine import cosine_similarity


MOOD_LABELS_PATH = Path("cache/mood_labels.pkl")
EMBEDDING_MODEL = "text-embedding-3-small"

MOOD_CATEGORIES: dict[str, str] = {
    "새벽 감성": "고요한 새벽, 늦은 밤, 몽환적이고 감성적인 공기, 차가운 야경과 어울리는 조용한 음악",
    "비 오는 날": "비 오는 창가, 빗소리, 촉촉하고 차분한 분위기, 우산을 쓰고 걷는 감성적인 음악",
    "그리움·이별": "보고 싶은 마음, 추억, 이별 뒤의 허전함, 혼자 남은 밤에 떠오르는 감정",
    "사랑·로맨스": "설렘, 사랑, 두근거림, 좋아하는 사람을 떠올릴 때의 달콤한 감정",
    "위로·힐링": "지친 마음을 달래는 위로, 따뜻함, 편안함, 천천히 숨을 고르게 하는 음악",
    "신남·파티": "에너지 넘치는 파티, 댄스, 신나는 비트, 흥겨움, 텐션을 끌어올리는 음악",
    "집중·공부": "카페에서 흐르는 잔잔한 배경음악, 로파이, 차분하지만 집중이 잘 되는 분위기, 공부나 작업에 어울리는 음악",
    "여행·자유": "드라이브, 여행, 탁 트인 길, 자유로운 기분, 바람을 맞으며 떠나고 싶은 음악",
}

MOOD_HINTS: dict[str, list[tuple[str, float]]] = {
    "집중·공부": [
        ("카페", 0.05),
        ("커피", 0.04),
        ("집중", 0.06),
        ("공부", 0.06),
        ("작업", 0.05),
        ("업무", 0.05),
        ("독서", 0.05),
        ("로파이", 0.05),
        ("lofi", 0.05),
        ("브금", 0.04),
        ("배경음악", 0.04),
        ("잔잔", 0.04),
        ("차분", 0.04),
    ],
    "신남·파티": [
        ("신나", 0.06),
        ("파티", 0.08),
        ("댄스", 0.06),
        ("춤", 0.05),
        ("클럽", 0.07),
        ("흥겨", 0.06),
        ("텐션", 0.05),
        ("업템포", 0.05),
        ("에너지", 0.05),
        ("축제", 0.05),
    ],
    "새벽 감성": [
        ("새벽", 0.06),
        ("밤", 0.04),
        ("야경", 0.04),
        ("심야", 0.05),
        ("몽환", 0.05),
    ],
    "비 오는 날": [
        ("비", 0.06),
        ("빗소리", 0.06),
        ("장마", 0.05),
        ("우산", 0.04),
        ("촉촉", 0.04),
    ],
    "위로·힐링": [
        ("위로", 0.07),
        ("힐링", 0.07),
        ("편안", 0.05),
        ("휴식", 0.05),
        ("쉼", 0.04),
        ("안정", 0.05),
    ],
    "여행·자유": [
        ("여행", 0.07),
        ("드라이브", 0.07),
        ("자유", 0.06),
        ("산책", 0.04),
        ("바다", 0.04),
    ],
    "사랑·로맨스": [
        ("사랑", 0.07),
        ("로맨스", 0.07),
        ("설렘", 0.06),
        ("썸", 0.06),
        ("연애", 0.05),
    ],
    "그리움·이별": [
        ("그리움", 0.07),
        ("이별", 0.08),
        ("추억", 0.05),
        ("보고 싶", 0.05),
        ("헤어", 0.05),
    ],
}


def load_or_create_mood_embeddings(client: OpenAI) -> dict[str, np.ndarray]:
    if MOOD_LABELS_PATH.exists():
        with open(MOOD_LABELS_PATH, "rb") as f:
            return pickle.load(f)

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

    return mood_embeddings


def classify_mood(query_vec: np.ndarray, mood_embeddings: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    scores = []
    for category, label_vec in mood_embeddings.items():
        score = cosine_similarity(query_vec, label_vec)
        scores.append((category, score))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def rerank_mood_ranking(query: str, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
    text = (query or "").strip().lower()
    has_cafe_context = "카페" in text or "cafe" in text
    has_party_intent = any(keyword in text for keyword in ["신나", "파티", "댄스", "클럽", "축제", "에너지"])
    adjusted = []
    for category, score in ranked:
        bonus = 0.0
        for keyword, weight in MOOD_HINTS.get(category, []):
            if keyword in text:
                bonus += weight
        if has_cafe_context and not has_party_intent:
            if category == "집중·공부":
                bonus += 0.12
            elif category == "신남·파티":
                bonus -= 0.04
        adjusted.append((category, score + min(bonus, 0.12)))
    return sorted(adjusted, key=lambda item: item[1], reverse=True)


def get_top_mood(query: str, client: OpenAI | None = None) -> tuple[str, float, list[tuple[str, float]]]:
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    mood_embeddings = load_or_create_mood_embeddings(client)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_vec = np.array(response.data[0].embedding)

    ranked = classify_mood(query_vec, mood_embeddings)
    ranked = rerank_mood_ranking(query, ranked)
    top_category, top_score = ranked[0]

    return top_category, top_score, ranked
