import argparse
import json

from .benchmark import flatten_benchmark_summary
from .comparison import build_backend_comparison_rows
from .filters import build_metadata_filter
from .data import write_expanded_dataset
from .services import (
    build_vector_map,
    compare_all_backends,
    delete_backend_records,
    get_backend_record,
    query_backend,
    run_benchmark,
    run_chroma_update_vs_upsert_demo,
    run_hybrid_demo,
    run_pinecone_idempotency_demo,
    sync_backend,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync")
    sync_parser.add_argument("--backend", choices=["chroma", "pinecone", "both"], default="both")
    sync_parser.add_argument("--size", type=int, default=None)
    sync_parser.add_argument("--force-upsert", action="store_true")

    hybrid_parser = subparsers.add_parser("hybrid")
    hybrid_parser.add_argument("--backend", choices=["chroma", "pinecone"], required=True)
    hybrid_parser.add_argument("--top-k", type=int, default=5)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--query", required=True)
    compare_parser.add_argument("--top-k", type=int, default=5)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--query", required=True)
    benchmark_parser.add_argument("--runs", type=int, default=3)

    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("--backend", choices=["chroma", "pinecone"], required=True)
    get_parser.add_argument("--id", required=True)
    get_parser.add_argument("--scope", default="all")
    get_parser.add_argument("--size", type=int, default=None)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--backend", choices=["chroma", "pinecone"], required=True)
    query_parser.add_argument("--query", required=True)
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--scope", default="all")
    query_parser.add_argument("--size", type=int, default=None)
    query_parser.add_argument("--genre", default="전체")
    query_parser.add_argument("--primary-mood", default="전체")
    query_parser.add_argument("--min-lyrics-length", type=int, default=0)
    query_parser.add_argument("--exclude-genre", default="없음")

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("--backend", choices=["chroma", "pinecone"], required=True)
    delete_parser.add_argument("--id", default=None)
    delete_parser.add_argument("--scope", default="all")
    delete_parser.add_argument("--size", type=int, default=None)
    delete_parser.add_argument("--genre", default="전체")
    delete_parser.add_argument("--primary-mood", default="전체")
    delete_parser.add_argument("--min-lyrics-length", type=int, default=0)
    delete_parser.add_argument("--exclude-genre", default="없음")

    subparsers.add_parser("compare-matrix")

    map_parser = subparsers.add_parser("map")
    map_parser.add_argument("--backend", choices=["chroma", "pinecone"], required=True)
    map_parser.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    map_parser.add_argument("--color-by", choices=["genre", "primary_mood"], default="genre")

    subparsers.add_parser("chroma-demo")
    subparsers.add_parser("pinecone-demo")

    expand_parser = subparsers.add_parser("expand")
    expand_parser.add_argument("--target-size", type=int, default=1000)

    args = parser.parse_args()

    if args.command == "sync":
        print(
            json.dumps(
                sync_backend(args.backend, size=args.size, skip_existing=not args.force_upsert),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "hybrid":
        print(json.dumps(run_hybrid_demo(args.backend, top_k=args.top_k), ensure_ascii=False, indent=2))
        return

    if args.command == "compare":
        print(json.dumps(compare_all_backends(args.query, top_k=args.top_k), ensure_ascii=False, indent=2))
        return

    if args.command == "benchmark":
        rows, summary, plot_path = run_benchmark(args.query, runs=args.runs)
        print(
            json.dumps(
                {"rows": rows, "summary": flatten_benchmark_summary(summary), "plot_path": plot_path},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "get":
        print(
            json.dumps(
                get_backend_record(args.backend, record_id=args.id, scope=args.scope, size=args.size),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "query":
        filter_expr = build_metadata_filter(
            genre=args.genre,
            primary_mood=args.primary_mood,
            min_lyrics_length=args.min_lyrics_length,
            exclude_genre=args.exclude_genre,
        )
        print(
            json.dumps(
                query_backend(
                    args.backend,
                    query=args.query,
                    top_k=args.top_k,
                    scope=args.scope,
                    metadata_filter=filter_expr,
                    size=args.size,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "delete":
        print(
            json.dumps(
                delete_backend_records(
                    backend=args.backend,
                    record_id=args.id,
                    scope=args.scope,
                    size=args.size,
                    genre=args.genre,
                    primary_mood=args.primary_mood,
                    min_lyrics_length=args.min_lyrics_length,
                    exclude_genre=args.exclude_genre,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "compare-matrix":
        print(json.dumps(build_backend_comparison_rows(), ensure_ascii=False, indent=2))
        return

    if args.command == "map":
        path, _ = build_vector_map(args.backend, method=args.method, color_by=args.color_by)
        print(json.dumps({"plot_path": path}, ensure_ascii=False, indent=2))
        return

    if args.command == "chroma-demo":
        print(json.dumps(run_chroma_update_vs_upsert_demo(), ensure_ascii=False, indent=2))
        return

    if args.command == "pinecone-demo":
        print(json.dumps(run_pinecone_idempotency_demo(), ensure_ascii=False, indent=2))
        return

    if args.command == "expand":
        path = write_expanded_dataset(target_size=args.target_size)
        print(json.dumps({"dataset_path": str(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
