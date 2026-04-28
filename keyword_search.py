"""
TF-IDF 키워드 검색 모듈 (시맨틱 검색 비교용)

동일 쿼리에 대해 시맨틱 검색과 키워드 검색 결과를 비교 분석한다.
"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


DATA_PATH = Path("data/songs.json")


def load_songs() -> list[dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_document_corpus(songs: list[dict]) -> list[str]:
    """검색 대상 문서: 가사 + 제목 + 아티스트 + 무드태그"""
    corpus = []
    for s in songs:
        mood_str = " ".join(s.get("mood_tags", []))
        doc = f"{s['title']} {s['artist']} {mood_str} {s['lyrics']}"
        corpus.append(doc)
    return corpus


def keyword_search(query: str, top_k: int = 5) -> list[dict]:
    """
    TF-IDF 기반 키워드 검색을 수행한다.

    Args:
        query: 검색 쿼리 문자열
        top_k: 반환할 결과 수

    Returns:
        TF-IDF 유사도 기준 상위 K개 곡 정보 목록
    """
    songs = load_songs()
    corpus = build_document_corpus(songs)

    # 쿼리를 corpus에 포함하여 TF-IDF 피처 공간 구성
    all_docs = corpus + [query]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",  # 한국어 지원: 문자 n-gram
        ngram_range=(2, 4),
        min_df=1,
        max_features=50000,
    )
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    query_vec = tfidf_matrix[-1]
    doc_matrix = tfidf_matrix[:-1]

    # 코사인 유사도 계산
    similarities = sklearn_cosine(query_vec, doc_matrix).flatten()
    sorted_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(sorted_indices):
        song = songs[idx].copy()
        song["tfidf_similarity"] = float(similarities[idx])
        song["rank"] = rank + 1

        # 쿼리와 가사 사이의 공통 키워드 추출
        query_tokens = set(re.findall(r"\w+", query))
        lyrics_tokens = set(re.findall(r"\w+", song["lyrics"]))
        common = query_tokens & lyrics_tokens
        song["common_keywords"] = list(common)
        song["keyword_overlap_count"] = len(common)

        results.append(song)

    return results


def compare_search_results(
    semantic_results: list[dict],
    keyword_results: list[dict],
    query: str,
) -> dict:
    """
    시맨틱 검색과 키워드 검색 결과를 비교하여 분석 데이터를 반환한다.
    """
    semantic_ids = {r["id"] for r in semantic_results}
    keyword_ids = {r["id"] for r in keyword_results}

    overlap = semantic_ids & keyword_ids
    semantic_only = semantic_ids - keyword_ids
    keyword_only = keyword_ids - semantic_ids

    query_tokens = set(re.findall(r"\w+", query))
    zero_overlap_semantic = [
        r for r in semantic_results
        if not (query_tokens & set(re.findall(r"\w+", r["lyrics"])))
    ]

    return {
        "query": query,
        "overlap_count": len(overlap),
        "semantic_only": list(semantic_only),
        "keyword_only": list(keyword_only),
        "zero_overlap_semantic_results": zero_overlap_semantic,
    }


if __name__ == "__main__":
    test_queries = [
        "비 오는 날 혼자 듣기 좋은 노래",
        "취업 준비에 지쳐서 위로받고 싶을 때",
        "새벽 드라이브에 신나게 달릴 때",
    ]

    for query in test_queries:
        print(f'\n쿼리: "{query}"')
        results = keyword_search(query, top_k=5)
        for r in results:
            kw = ", ".join(r["common_keywords"]) if r["common_keywords"] else "(없음)"
            print(f"  {r['rank']}. {r['title']} - {r['artist']} "
                  f"(TF-IDF: {r['tfidf_similarity']:.4f}, 공통 키워드: {kw})")
