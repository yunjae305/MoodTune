import time

import numpy as np

from search import search_from_query_vector
from .benchmark import (
    benchmark_file_search,
    build_scaled_cache,
    load_base_cache,
    plot_benchmark_summary,
    summarize_benchmark_rows,
)
from .data import build_vector_records, expand_song_dataset, load_song_dataset
from .filters import build_metadata_filter
from .store import ChromaSongStore, OpenAIEmbedder, PineconeSongStore, group_records_by_scope
from .visualizer import plot_vector_map
from .workflows import (
    build_benchmark_datasets,
    build_chroma_hybrid_scenarios,
    build_pinecone_hybrid_scenarios,
)


def dataset_name_for_size(size: int | None) -> str:
    if size is None:
        return "base"
    return f"scale_{size}"


def load_songs_for_size(size: int | None = None) -> list[dict]:
    songs = load_song_dataset()
    if size is None:
        return songs
    return build_benchmark_datasets(songs, scales=(size,))[size]


def build_records_for_size(size: int | None = None, tenant: str = "main") -> list[dict]:
    songs = load_songs_for_size(size=size)
    return build_vector_records(songs, tenant=tenant, dataset_name=dataset_name_for_size(size))


def create_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder()


def create_chroma_store(size: int | None = None) -> ChromaSongStore:
    suffix = dataset_name_for_size(size)
    return ChromaSongStore(collection_prefix=f"moodtune_{suffix}")


def create_pinecone_store(size: int | None = None) -> PineconeSongStore:
    suffix = dataset_name_for_size(size)
    return PineconeSongStore(namespace_prefix=suffix)


def sync_backend(backend: str, size: int | None = None, skip_existing: bool = True) -> dict:
    """DB 초기화 단계에서 이미 저장된 ID를 재임베딩하지 않도록 동기화한다."""
    records = build_records_for_size(size=size)
    if backend == "chroma":
        return create_chroma_store(size=size).sync_records(records, skip_existing=skip_existing)
    if backend == "pinecone":
        return create_pinecone_store(size=size).sync_records(records, skip_existing=skip_existing)
    if backend == "both":
        return {
            "chroma": create_chroma_store(size=size).sync_records(records, skip_existing=skip_existing),
            "pinecone": create_pinecone_store(size=size).sync_records(records, skip_existing=skip_existing),
        }
    raise ValueError(f"unsupported backend: {backend}")


def query_backend(
    backend: str,
    query: str,
    top_k: int = 5,
    scope: str = "all",
    metadata_filter: dict | None = None,
    size: int | None = None,
) -> list[dict]:
    if backend == "chroma":
        return create_chroma_store(size=size).query(query, top_k=top_k, scope=scope, where=metadata_filter)
    if backend == "pinecone":
        return create_pinecone_store(size=size).query(query, top_k=top_k, scope=scope, filter=metadata_filter)
    raise ValueError(f"unsupported backend: {backend}")


def get_backend_record(backend: str, record_id: str, scope: str = "all", size: int | None = None) -> dict:
    if backend == "chroma":
        payload = create_chroma_store(size=size).get_by_id(record_id, scope=scope)
        ids = payload.get("ids", [])
        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])
        if not ids:
            return {"backend": backend, "scope": scope, "found": False, "id": record_id}
        row = dict((metadatas or [{}])[0] or {})
        row["id"] = ids[0]
        row["document"] = (documents or [""])[0]
        return {"backend": backend, "scope": scope, "found": True, "record": row}
    if backend == "pinecone":
        payload = create_pinecone_store(size=size).get_by_id(record_id, scope=scope)
        if hasattr(payload, "vectors"):
            vectors = payload.vectors
        else:
            vectors = payload.get("vectors", {})
        vector = vectors.get(record_id)
        if vector is None:
            return {"backend": backend, "scope": scope, "found": False, "id": record_id}
        metadata = getattr(vector, "metadata", None)
        values = getattr(vector, "values", None)
        return {
            "backend": backend,
            "scope": scope,
            "found": True,
            "record": {
                "id": record_id,
                "metadata": metadata,
                "dimension": len(values or []),
            },
        }
    raise ValueError(f"unsupported backend: {backend}")


def delete_backend_records(
    backend: str,
    record_id: str | None = None,
    scope: str = "all",
    size: int | None = None,
    genre: str = "전체",
    primary_mood: str = "전체",
    min_lyrics_length: int = 0,
    exclude_genre: str = "없음",
) -> dict:
    """ID 삭제와 조건 삭제를 같은 인터페이스로 호출할 수 있게 한다."""
    metadata_filter = build_metadata_filter(
        genre=genre,
        primary_mood=primary_mood,
        min_lyrics_length=min_lyrics_length,
        exclude_genre=exclude_genre,
    )
    if record_id is None and metadata_filter is None:
        raise ValueError("record_id or metadata filter is required")
    if backend == "chroma":
        store = create_chroma_store(size=size)
        before = get_backend_record(backend, record_id, scope=scope, size=size) if record_id else None
        store.delete_records(scope=scope, ids=[record_id] if record_id else None, where=metadata_filter)
        after = get_backend_record(backend, record_id, scope=scope, size=size) if record_id else None
        return {"backend": backend, "scope": scope, "before": before, "after": after, "filter": metadata_filter}
    if backend == "pinecone":
        store = create_pinecone_store(size=size)
        before = get_backend_record(backend, record_id, scope=scope, size=size) if record_id else None
        store.delete_records(scope=scope, ids=[record_id] if record_id else None, filter=metadata_filter)
        after = get_backend_record(backend, record_id, scope=scope, size=size) if record_id else None
        return {"backend": backend, "scope": scope, "before": before, "after": after, "filter": metadata_filter}
    raise ValueError(f"unsupported backend: {backend}")


def query_file_backend(query: str, top_k: int = 5, size: int | None = None) -> list[dict]:
    base_cache = load_base_cache()
    songs = load_songs_for_size(size=size)
    if size is None:
        cache = base_cache
    else:
        cache = build_scaled_cache(base_cache, songs)
    query_vector = np.array(create_embedder().embed_text(query))
    return search_from_query_vector(query_vec=query_vector, cache=cache, top_k=top_k)


def run_hybrid_demo(backend: str, top_k: int = 5, size: int | None = None) -> list[dict]:
    scenarios = build_chroma_hybrid_scenarios() if backend == "chroma" else build_pinecone_hybrid_scenarios()
    rows = []
    for scenario in scenarios:
        pure = query_backend(backend, scenario["query"], top_k=top_k, size=size)
        filter_payload = scenario.get("where") if backend == "chroma" else scenario.get("filter")
        hybrid = query_backend(
            backend,
            scenario["query"],
            top_k=top_k,
            metadata_filter=filter_payload,
            size=size,
        )
        rows.append(
            {
                "name": scenario["name"],
                "query": scenario["query"],
                "filter": filter_payload,
                "pure": pure,
                "hybrid": hybrid,
            }
        )
    return rows


def run_chroma_update_vs_upsert_demo(scope: str = "all") -> dict:
    """update는 누락 ID를 만들지 않고 upsert는 새 레코드를 만든다는 차이를 보여준다."""
    store = ChromaSongStore(collection_prefix="moodtune_demo")
    missing_id = "demo_missing_record"
    store.delete_records(scope=scope, ids=[missing_id])
    before = store.get_by_id(missing_id, scope=scope)
    store.update_record(
        missing_id,
        scope=scope,
        document="update only document",
        metadata={"genre": "demo"},
    )
    after_update = store.get_by_id(missing_id, scope=scope)
    record = {
        "id": missing_id,
        "document": "upserted document",
        "metadata": {
            "song_id": missing_id,
            "title": "Demo Upsert",
            "artist": "MoodTune",
            "genre": "demo",
            "genre_key": "genre_demo",
            "primary_mood": "demo",
            "mood_tags": ["demo"],
            "lyrics_length": 16,
            "title_length": 11,
            "mood_count": 1,
            "tenant": "main",
            "dataset_name": "base",
            "variant_index": 0,
            "source_song_id": missing_id,
            "has_multiple_moods": False,
        },
    }
    store.sync_records([record], skip_existing=False)
    after_upsert = store.get_by_id(missing_id, scope=scope)
    return {
        "before_ids": before.get("ids", []),
        "after_update_ids": after_update.get("ids", []),
        "after_upsert_ids": after_upsert.get("ids", []),
        "note": "update ignores a missing id, while upsert inserts it",
    }


def run_pinecone_idempotency_demo(scope: str = "all") -> dict:
    """같은 upsert를 반복해도 namespace 상태가 변하지 않는지 확인한다."""
    store = PineconeSongStore(namespace_prefix="demo")
    records = build_records_for_size(size=None)[:5]
    store.sync_records(records, skip_existing=False)
    first_ids = store.list_ids(scope=scope)
    store.sync_records(records, skip_existing=False)
    second_ids = store.list_ids(scope=scope)
    sample = store.get_by_id(records[0]["id"], scope=scope)
    if hasattr(sample, "vectors"):
        vectors = sample.vectors
    else:
        vectors = sample.get("vectors", {})
    first_vector = next(iter(vectors.values())) if vectors else None
    sample = {
        "namespace": getattr(sample, "namespace", scope),
        "vector_ids": list(vectors.keys()),
        "sample_metadata": getattr(first_vector, "metadata", None),
    }
    return {
        "first_count": len(first_ids),
        "second_count": len(second_ids),
        "same_ids": sorted(first_ids) == sorted(second_ids),
        "sample": sample,
        "note": "repeated upserts keep the namespace content stable",
    }


def compare_all_backends(query: str, top_k: int = 5, size: int | None = None) -> dict:
    return {
        "file": query_file_backend(query, top_k=top_k, size=size),
        "chroma": query_backend("chroma", query, top_k=top_k, size=size),
        "pinecone": query_backend("pinecone", query, top_k=top_k, size=size),
    }


def fetch_visualization_payload(backend: str, scope: str = "all", size: int | None = None) -> tuple[np.ndarray, list[dict]]:
    if backend == "chroma":
        payload = create_chroma_store(size=size).fetch_all(scope=scope)
        embeddings = np.array(payload.get("embeddings", []))
        rows = payload.get("metadatas", [])
        return embeddings, rows
    if backend == "pinecone":
        payload = create_pinecone_store(size=size).fetch_all(scope=scope)
        embeddings = np.array([row.get("values", []) for row in payload])
        rows = [row.get("metadata", {}) for row in payload]
        return embeddings, rows
    raise ValueError(f"unsupported backend: {backend}")


def build_vector_map(
    backend: str,
    method: str = "tsne",
    scope: str = "all",
    color_by: str = "genre",
    size: int | None = None,
) -> tuple[str, np.ndarray]:
    embeddings, rows = fetch_visualization_payload(backend, scope=scope, size=size)
    path, coords = plot_vector_map(
        embeddings=embeddings,
        rows=rows,
        method=method,
        color_by=color_by,
        title=f"{backend.upper()} {method.upper()} Map",
    )
    return str(path), coords


def benchmark_scopes(records: list[dict]) -> list[str]:
    return list(group_records_by_scope(records).keys())


def is_missing_chroma_segment_error(error: Exception) -> bool:
    message = str(error).lower()
    return "nothing found on disk" in message and "segment" in message


def run_benchmark(query: str, sizes: tuple[int, ...] = (100, 500, 1000), top_k: int = 5, runs: int = 3) -> tuple[list[dict], dict, str]:
    """파일 기반 검색과 두 벡터 DB를 같은 질의와 같은 스케일로 비교한다."""
    rows = []
    base_cache = load_base_cache()
    base_songs = load_song_dataset()
    datasets = build_benchmark_datasets(base_songs, scales=sizes)
    embedder = create_embedder()
    query_vector = np.array(embedder.embed_text(query))

    for size in sizes:
        songs = datasets[size]
        records = build_vector_records(songs, tenant="benchmark", dataset_name=dataset_name_for_size(size))
        scopes = benchmark_scopes(records)
        chroma_store = create_chroma_store(size=size)
        pinecone_store = create_pinecone_store(size=size)
        chroma_store.reset_scopes(scopes)
        pinecone_store.reset_scopes(scopes)
        chroma_store.sync_records(records, skip_existing=False)
        pinecone_store.sync_records(records, skip_existing=False)

        scaled_cache = build_scaled_cache(base_cache, songs)
        for latency in benchmark_file_search(scaled_cache, query_vector=query_vector, top_k=top_k, runs=runs):
            rows.append({"backend": "file", "size": size, "latency_ms": latency})

        for _ in range(runs):
            start = time.perf_counter()
            try:
                chroma_store.query(query, top_k=top_k)
            except Exception as error:
                if not is_missing_chroma_segment_error(error):
                    raise
                chroma_store.reset_scopes(scopes)
                chroma_store.sync_records(records, skip_existing=False)
                chroma_store.query(query, top_k=top_k)
            rows.append({"backend": "chroma", "size": size, "latency_ms": (time.perf_counter() - start) * 1000})

        for _ in range(runs):
            start = time.perf_counter()
            pinecone_store.query(query, top_k=top_k)
            rows.append({"backend": "pinecone", "size": size, "latency_ms": (time.perf_counter() - start) * 1000})

    summary = summarize_benchmark_rows(rows)
    plot_path = plot_benchmark_summary(summary)
    return rows, summary, str(plot_path)
