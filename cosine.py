"""
코사인 유사도 직접 구현 및 scipy 검증 모듈

수식: cos(θ) = (A · B) / (||A|| × ||B||)
"""

import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    두 벡터 간 코사인 유사도를 직접 계산한다.
    반환값 범위: -1.0 (정반대) ~ 1.0 (동일 방향)
    """
    # 내적 계산: A · B = Σ(aᵢ × bᵢ)
    dot_product = np.dot(vec_a, vec_b)

    # 벡터 크기(L2 노름) 계산: ||A|| = √(Σaᵢ²)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # 분모가 0이면 (영벡터) 유사도는 0
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def cosine_similarity_batch(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    쿼리 벡터와 임베딩 행렬 전체에 대한 유사도를 일괄 계산한다.
    matrix shape: (N, D) — N개 곡, D차원 임베딩
    """
    # 쿼리 벡터 크기
    query_norm = np.linalg.norm(query_vec)

    # 행렬의 각 행(곡)에 대한 크기
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # 분모: ||query|| × ||각 곡 벡터||
    denominators = query_norm * matrix_norms

    # 내적을 행렬 연산으로 일괄 계산
    dot_products = matrix @ query_vec

    # 분모가 0인 경우 처리
    with np.errstate(divide='ignore', invalid='ignore'):
        similarities = np.where(denominators != 0, dot_products / denominators, 0.0)

    return similarities


def verify_against_scipy(vec_a: np.ndarray, vec_b: np.ndarray, tolerance: float = 1e-6) -> dict:
    """
    직접 구현한 코사인 유사도와 scipy 결과를 비교 검증한다.
    scipy는 '거리(distance)'를 반환하므로 유사도 = 1 - 거리
    """
    our_result = cosine_similarity(vec_a, vec_b)
    scipy_distance = scipy_cosine(vec_a, vec_b)
    scipy_result = 1.0 - scipy_distance

    error = abs(our_result - scipy_result)
    is_valid = error < tolerance

    return {
        "our_similarity": our_result,
        "scipy_similarity": scipy_result,
        "error": error,
        "is_valid": is_valid,
        "tolerance": tolerance,
    }


def demo_similarity_comparison(embeddings: list[np.ndarray], song_titles: list[str]) -> None:
    """
    유사한 곡 쌍 vs 유사하지 않은 곡 쌍의 거리 수치를 출력한다.
    """
    if len(embeddings) < 4:
        print("비교를 위해 최소 4개 이상의 임베딩이 필요합니다.")
        return

    print("=" * 60)
    print("코사인 유사도 비교 (직접 구현 vs scipy)")
    print("=" * 60)

    pairs_to_compare = [
        (0, 1, "유사한 쌍"),
        (0, len(embeddings) // 2, "다소 다른 쌍"),
        (0, len(embeddings) - 1, "유사하지 않은 쌍"),
    ]

    for idx_a, idx_b, label in pairs_to_compare:
        result = verify_against_scipy(embeddings[idx_a], embeddings[idx_b])
        print(f"\n[{label}]")
        print(f"  곡 A: {song_titles[idx_a]}")
        print(f"  곡 B: {song_titles[idx_b]}")
        print(f"  직접 구현 유사도: {result['our_similarity']:.6f}")
        print(f"  scipy 유사도:    {result['scipy_similarity']:.6f}")
        print(f"  오차: {result['error']:.2e}  ({'검증 통과 ✓' if result['is_valid'] else '검증 실패 ✗'})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("코사인 유사도 직접 구현 검증\n")

    # 간단한 테스트 벡터로 수식 동작 확인
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    d = np.array([-1.0, 0.0, 0.0])

    print(f"동일 벡터 (기대값 1.0): {cosine_similarity(a, b):.6f}")
    print(f"직교 벡터 (기대값 0.0): {cosine_similarity(a, c):.6f}")
    print(f"반대 벡터 (기대값 -1.0): {cosine_similarity(a, d):.6f}")

    print("\n--- scipy 검증 ---")
    for vec_pair, desc in [((a, b), "동일"), ((a, c), "직교"), ((a, d), "반대")]:
        result = verify_against_scipy(*vec_pair)
        status = "✓" if result["is_valid"] else "✗"
        print(f"{desc}: 오차 = {result['error']:.2e} {status}")

    # 랜덤 고차원 벡터 검증 (실제 임베딩 크기 시뮬레이션)
    print("\n--- 1536차원 랜덤 벡터 검증 ---")
    rng = np.random.default_rng(42)
    v1 = rng.standard_normal(1536)
    v2 = rng.standard_normal(1536)
    result = verify_against_scipy(v1, v2)
    status = "✓" if result["is_valid"] else "✗"
    print(f"랜덤 벡터 쌍: 직접={result['our_similarity']:.6f}, scipy={result['scipy_similarity']:.6f}, 오차={result['error']:.2e} {status}")
