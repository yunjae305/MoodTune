"""
t-SNE 임베딩 시각화 모듈

100곡 임베딩을 2D로 축소하여 무드 카테고리별 클러스터를 시각화한다.
"""

import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from sklearn.manifold import TSNE


CACHE_DIR = Path("cache")

# 무드별 색상 매핑
MOOD_COLOR_MAP = {
    "새벽": "#3B4CCA",
    "비": "#6495ED",
    "그리움": "#9370DB",
    "이별": "#8B008B",
    "설렘": "#FF69B4",
    "사랑": "#FF1493",
    "위로": "#32CD32",
    "힐링": "#228B22",
    "신남": "#FF8C00",
    "파티": "#FF4500",
    "집중": "#708090",
    "공부": "#2F4F4F",
    "여행": "#00CED1",
    "자유": "#20B2AA",
    "감성적": "#DDA0DD",
    "잔잔한": "#B0C4DE",
}

DEFAULT_COLOR = "#AAAAAA"


def get_song_primary_mood(mood_tags: list[str]) -> str:
    """곡의 첫 번째 무드 태그를 대표 무드로 반환한다."""
    for tag in mood_tags:
        if tag in MOOD_COLOR_MAP:
            return tag
    return "기타"


def load_embeddings(enriched: bool = True) -> tuple[np.ndarray, list[dict]]:
    path = CACHE_DIR / ("enriched_embeddings.pkl" if enriched else "embeddings.pkl")
    if not path.exists():
        raise FileNotFoundError(f"캐시 파일 없음: {path}\n`python embed_songs.py`를 먼저 실행하세요.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(data["embeddings"]), data["songs"]


def run_tsne(embeddings: np.ndarray, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
    """t-SNE로 고차원 임베딩을 2D로 축소한다."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
        learning_rate="auto",
        init="pca",
    )
    return tsne.fit_transform(embeddings)


def visualize(
    query_vec_2d: np.ndarray | None = None,
    query_label: str = "검색 쿼리",
    enriched: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    t-SNE 산점도를 생성한다.

    Args:
        query_vec_2d: t-SNE 공간에서의 쿼리 벡터 2D 좌표 (있으면 빨간 별로 표시)
        query_label: 쿼리 레이블 텍스트
        enriched: 풍부한 임베딩 사용 여부
        save_path: 이미지 저장 경로 (None이면 저장 안 함)

    Returns:
        matplotlib Figure 객체
    """
    # 한국어 폰트 설정
    _setup_korean_font()

    embeddings, songs = load_embeddings(enriched=enriched)
    print(f"t-SNE 실행 중 (perplexity=30, n_iter=1000)...")
    coords_2d = run_tsne(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # 무드별로 그룹화하여 플롯
    plotted_moods = set()
    for i, (song, (x, y)) in enumerate(zip(songs, coords_2d)):
        primary_mood = get_song_primary_mood(song.get("mood_tags", []))
        color = MOOD_COLOR_MAP.get(primary_mood, DEFAULT_COLOR)

        label = primary_mood if primary_mood not in plotted_moods else None
        ax.scatter(x, y, c=color, s=80, alpha=0.85, label=label,
                   edgecolors="white", linewidths=0.5)
        plotted_moods.add(primary_mood)

        ax.annotate(
            f"{song['title'][:6]}",
            (x, y),
            fontsize=5.5,
            color="white",
            alpha=0.7,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    # 쿼리 벡터를 빨간 별로 표시
    if query_vec_2d is not None:
        ax.scatter(
            query_vec_2d[0], query_vec_2d[1],
            c="#FF0000", s=300, marker="*",
            zorder=10, label=f"★ {query_label}",
            edgecolors="white", linewidths=1.5,
        )

    ax.set_title("MoodTune — 가사 임베딩 t-SNE 시각화\n(무드 카테고리별 색상 구분)",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("t-SNE 1차원", color="white", fontsize=10)
    ax.set_ylabel("t-SNE 2차원", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    legend = ax.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.3,
        facecolor="#0f3460",
        edgecolor="#444466",
        labelcolor="white",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"시각화 저장: {save_path}")

    return fig


def get_tsne_coords_for_query(
    query_vec: np.ndarray,
    enriched: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    쿼리 벡터를 기존 임베딩과 함께 t-SNE 공간에 투영한다.
    쿼리를 마지막 벡터로 추가하여 동일 공간에서 축소한다.
    """
    embeddings, songs = load_embeddings(enriched=enriched)
    all_vecs = np.vstack([embeddings, query_vec.reshape(1, -1)])
    coords = run_tsne(all_vecs)
    return coords[:-1], coords[-1]  # (곡들 좌표, 쿼리 좌표)


def _setup_korean_font() -> None:
    """시스템에서 한국어 폰트를 찾아 matplotlib에 설정한다."""
    korean_fonts = [
        "Malgun Gothic",    # Windows
        "Apple SD Gothic Neo",  # macOS
        "NanumGothic",      # Linux
        "NanumBarunGothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in korean_fonts:
        if font in available:
            matplotlib.rc("font", family=font)
            break
    matplotlib.rc("axes", unicode_minus=False)


if __name__ == "__main__":
    fig = visualize(save_path="cache/tsne_plot.png")
    plt.show()
