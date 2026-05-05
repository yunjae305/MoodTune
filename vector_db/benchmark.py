import pickle
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from search import search_from_query_vector
from .data import EXPANDED_DATA_PATH, load_song_dataset
from .workflows import build_benchmark_datasets


ENRICHED_CACHE_PATH = Path("cache/enriched_embeddings.pkl")
BENCHMARK_PLOT_PATH = Path("cache/vector_benchmark.png")


def load_base_cache(path: Path | None = None) -> dict:
    cache_path = path or ENRICHED_CACHE_PATH
    with open(cache_path, "rb") as handle:
        return pickle.load(handle)


def build_scaled_cache(base_cache: dict, expanded_songs: list[dict]) -> dict:
    embedding_by_song_id = {
        song["id"]: embedding
        for song, embedding in zip(base_cache["songs"], base_cache["embeddings"])
    }
    songs = []
    embeddings = []
    for song in expanded_songs:
        source_song_id = song.get("source_song_id", song["id"])
        if source_song_id not in embedding_by_song_id:
            raise KeyError(f"missing source embedding for {source_song_id}")
        songs.append(song)
        embeddings.append(embedding_by_song_id[source_song_id])
    return {
        "model": base_cache.get("model", "text-embedding-3-small"),
        "songs": songs,
        "embeddings": embeddings,
    }


def benchmark_file_search(cache: dict, query_vector: np.ndarray, top_k: int = 5, runs: int = 5) -> list[float]:
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        search_from_query_vector(query_vec=query_vector, cache=cache, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def summarize_benchmark_rows(rows: list[dict]) -> dict[tuple[str, int], dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["backend"], row["size"])].append(row["latency_ms"])
    summary = {}
    for key, values in grouped.items():
        summary[key] = {
            "runs": len(values),
            "avg_latency_ms": sum(values) / len(values),
            "min_latency_ms": min(values),
            "max_latency_ms": max(values),
        }
    return summary


def flatten_benchmark_summary(summary: dict[tuple[str, int], dict]) -> list[dict]:
    rows = []
    for (backend, size), metrics in sorted(summary.items(), key=lambda item: (item[0][1], item[0][0])):
        row = {"backend": backend, "size": size}
        row.update(metrics)
        rows.append(row)
    return rows


def plot_benchmark_summary(summary: dict[tuple[str, int], dict], save_path: Path | None = None) -> Path:
    output_path = save_path or BENCHMARK_PLOT_PATH
    backend_names = sorted({backend for backend, _ in summary})
    sizes = sorted({size for _, size in summary})
    x = np.arange(len(sizes))
    width = 0.24 if backend_names else 0.4

    fig, ax = plt.subplots(figsize=(11, 6))
    for index, backend in enumerate(backend_names):
        values = [summary[(backend, size)]["avg_latency_ms"] for size in sizes]
        ax.bar(x + (index * width), values, width=width, label=backend)

    ax.set_title("Vector Search Benchmark")
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Average Latency (ms)")
    ax.set_xticks(x + width * max(len(backend_names) - 1, 0) / 2)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def ensure_expanded_dataset_file(target_size: int = 1000) -> Path:
    if EXPANDED_DATA_PATH.exists():
        return EXPANDED_DATA_PATH
    songs = load_song_dataset()
    datasets = build_benchmark_datasets(songs, scales=(target_size,))
    EXPANDED_DATA_PATH.write_text(
        __import__("json").dumps(datasets[target_size], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return EXPANDED_DATA_PATH
