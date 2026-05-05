import argparse
import json

from vector_db.services import sync_backend


def main() -> None:
    """과제 제출용 DB 초기화 스크립트다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["chroma", "pinecone", "both"], default="both")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--force-upsert", action="store_true")
    args = parser.parse_args()
    payload = sync_backend(args.backend, size=args.size, skip_existing=not args.force_upsert)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
