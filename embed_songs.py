"""
가사 임베딩 생성 및 pickle 캐싱 스크립트

실행: python embed_songs.py
"""

import json
import os
import pickle
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


DATA_PATH = Path("data/songs.json")
CACHE_DIR = Path("cache")
EMBEDDINGS_PATH = CACHE_DIR / "embeddings.pkl"
ENRICHED_EMBEDDINGS_PATH = CACHE_DIR / "enriched_embeddings.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_api_key(env_path: Path | None = None) -> str:
    load_dotenv(dotenv_path=env_path)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    return api_key


def load_songs() -> list[dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_enriched_text(song: dict) -> str:
    """제목 + 아티스트 + 무드태그 + 가사를 결합한 풍부한 임베딩 입력 텍스트"""
    mood_str = ", ".join(song.get("mood_tags", []))
    return (
        f"제목: {song['title']}\n"
        f"아티스트: {song['artist']}\n"
        f"무드: {mood_str}\n"
        f"장르: {song.get('genre', '')}\n"
        f"가사: {song['lyrics']}"
    )


def get_embeddings(client: OpenAI, texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """OpenAI 임베딩 API를 호출하여 벡터 목록을 반환한다."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  배치 {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} 처리 중... ({len(batch)}개)")

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        # API 속도 제한 방지
        if i + batch_size < len(texts):
            time.sleep(0.5)

    return all_embeddings


def save_cache(data: dict, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  캐시 저장 완료: {path} ({path.stat().st_size / 1024:.1f} KB)")


def load_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def is_cache_valid(cache: dict | None, songs: list[dict]) -> bool:
    if cache is None:
        return False
    cached_songs = cache.get("songs", [])
    cached_embeddings = cache.get("embeddings", [])
    if len(cached_songs) != len(songs) or len(cached_embeddings) != len(songs):
        return False
    return cached_songs == songs


def main() -> None:
    client = OpenAI(api_key=get_api_key())
    songs = load_songs()
    print(f"총 {len(songs)}곡 로드 완료\n")

    # --- 단순 임베딩 (가사 텍스트만) ---
    print("[1/2] 단순 임베딩 생성 (가사 텍스트)")
    simple_cache = load_cache(EMBEDDINGS_PATH)
    if is_cache_valid(simple_cache, songs):
        print("  캐시 파일 존재 - 건너뜀")
    else:
        lyrics_texts = [s["lyrics"] for s in songs]
        embeddings = get_embeddings(client, lyrics_texts)
        cache_data = {
            "model": EMBEDDING_MODEL,
            "songs": songs,
            "embeddings": embeddings,
        }
        save_cache(cache_data, EMBEDDINGS_PATH)
        print(f"  단순 임베딩 완료: {len(embeddings)}개")

    # --- 풍부한 임베딩 (제목 + 아티스트 + 무드태그 + 가사) ---
    print("\n[2/2] 풍부한 임베딩 생성 (제목+아티스트+무드+가사)")
    enriched_cache = load_cache(ENRICHED_EMBEDDINGS_PATH)
    if is_cache_valid(enriched_cache, songs):
        print("  캐시 파일 존재 - 건너뜀")
    else:
        enriched_texts = [build_enriched_text(s) for s in songs]
        enriched_embeddings = get_embeddings(client, enriched_texts)
        enriched_cache = {
            "model": EMBEDDING_MODEL,
            "songs": songs,
            "embeddings": enriched_embeddings,
        }
        save_cache(enriched_cache, ENRICHED_EMBEDDINGS_PATH)
        print(f"  풍부한 임베딩 완료: {len(enriched_embeddings)}개")

    print("\n모든 임베딩 캐싱 완료!")


if __name__ == "__main__":
    main()
