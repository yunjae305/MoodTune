import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


DATA_PATH = Path("data/songs.json")
EXPANDED_DATA_PATH = Path("data/songs_expanded_1000.json")

VARIANT_PROMPTS = [
    "late night session",
    "study room mix",
    "drive edition",
    "rain window cut",
    "soft playlist take",
    "weekend replay",
]


def load_song_dataset(path: Path | None = None) -> list[dict]:
    dataset_path = path or DATA_PATH
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def build_song_document(song: dict) -> str:
    mood_tags = ", ".join(song.get("mood_tags", []))
    return "\n".join(
        [
            f"title: {song.get('title', '')}",
            f"artist: {song.get('artist', '')}",
            f"genre: {song.get('genre', '')}",
            f"mood_tags: {mood_tags}",
            f"lyrics: {song.get('lyrics', '')}",
        ]
    )


def build_genre_key(genre: str) -> str:
    normalized = (genre or "unknown").strip()
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:8]
    return f"genre_{digest}"


def get_primary_mood(song: dict) -> str:
    mood_tags = song.get("mood_tags", [])
    if mood_tags:
        return mood_tags[0]
    return "unknown"


def build_song_metadata(
    song: dict,
    tenant: str = "main",
    dataset_name: str = "base",
    variant_index: int = 0,
) -> dict:
    lyrics = song.get("lyrics", "")
    genre = song.get("genre", "unknown")
    primary_mood = get_primary_mood(song)
    mood_tags = song.get("mood_tags", [])
    return {
        "song_id": song.get("id", ""),
        "title": song.get("title", ""),
        "artist": song.get("artist", ""),
        "genre": genre,
        "genre_key": build_genre_key(genre),
        "primary_mood": primary_mood,
        "mood_tags": mood_tags,
        "lyrics_length": len(lyrics),
        "title_length": len(song.get("title", "")),
        "mood_count": len(mood_tags),
        "tenant": tenant,
        "dataset_name": dataset_name,
        "variant_index": variant_index,
        "source_song_id": song.get("source_song_id", song.get("id", "")),
        "has_multiple_moods": len(mood_tags) > 1,
    }


def build_vector_records(
    songs: list[dict],
    tenant: str = "main",
    dataset_name: str = "base",
) -> list[dict]:
    """파일 기반 곡 데이터를 벡터 DB 적재용 문서와 메타데이터로 바꾼다."""
    records = []
    for song in songs:
        records.append(
            {
                "id": song["id"],
                "document": build_song_document(song),
                "metadata": build_song_metadata(
                    song,
                    tenant=tenant,
                    dataset_name=dataset_name,
                    variant_index=song.get("variant_index", 0),
                ),
            }
        )
    return records


def expand_song_dataset(songs: list[dict], target_size: int = 1000) -> list[dict]:
    if target_size <= len(songs):
        return copy.deepcopy(songs[:target_size])

    expanded = copy.deepcopy(songs)
    variant_counters = Counter()
    cursor = 0
    while len(expanded) < target_size:
        base_song = songs[cursor % len(songs)]
        cursor += 1
        variant_counters[base_song["id"]] += 1
        variant_index = variant_counters[base_song["id"]]
        variant = copy.deepcopy(base_song)
        variant["source_song_id"] = base_song["id"]
        variant["variant_index"] = variant_index
        variant["id"] = f"{base_song['id']}__variant_{variant_index:03d}"
        variant["title"] = f"{base_song['title']} Session {variant_index}"
        variant["lyrics"] = (
            f"{base_song.get('lyrics', '')}\n"
            f"playlist cut: {VARIANT_PROMPTS[(variant_index - 1) % len(VARIANT_PROMPTS)]}"
        )
        expanded.append(variant)
    return expanded


def write_expanded_dataset(path: Path | None = None, target_size: int = 1000) -> Path:
    """1000개 규모 벤치마크용 확장 데이터셋을 필요할 때만 생성한다."""
    output_path = path or EXPANDED_DATA_PATH
    songs = load_song_dataset()
    expanded = expand_song_dataset(songs, target_size=target_size)
    output_path.write_text(
        json.dumps(expanded, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
