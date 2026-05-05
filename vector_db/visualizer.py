from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


VECTOR_MAP_PATH = Path("cache/vector_map.png")


def build_color_labels(rows: list[dict], color_by: str = "genre") -> list[str]:
    labels = []
    for row in rows:
        labels.append(str(row.get(color_by) or "unknown"))
    return labels


def reduce_embeddings(
    embeddings: np.ndarray,
    method: str = "tsne",
    perplexity: int = 30,
    random_state: int = 42,
) -> np.ndarray:
    if method == "umap":
        import umap

        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.15, random_state=random_state)
        return reducer.fit_transform(embeddings)
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
        max_iter=1000,
    )
    return reducer.fit_transform(embeddings)


def plot_vector_map(
    embeddings: np.ndarray,
    rows: list[dict],
    method: str = "tsne",
    color_by: str = "genre",
    title: str = "Vector Map",
    save_path: Path | None = None,
) -> tuple[Path, np.ndarray]:
    coords = reduce_embeddings(embeddings, method=method, perplexity=max(2, min(30, len(embeddings) - 1)))
    labels = build_color_labels(rows, color_by=color_by)
    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    fig, ax = plt.subplots(figsize=(12, 7))
    for index, label in enumerate(unique_labels):
        indices = [row_index for row_index, row_label in enumerate(labels) if row_label == label]
        points = coords[indices]
        ax.scatter(points[:, 0], points[:, 1], s=42, alpha=0.8, label=label, color=colors(index))

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    output_path = save_path or VECTOR_MAP_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path, coords
