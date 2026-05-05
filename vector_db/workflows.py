from .data import expand_song_dataset


def build_benchmark_datasets(songs: list[dict], scales: tuple[int, ...] = (100, 500, 1000)) -> dict[int, list[dict]]:
    datasets = {}
    for scale in scales:
        datasets[scale] = expand_song_dataset(songs, target_size=scale)
    return datasets


def build_chroma_hybrid_scenarios() -> list[dict]:
    return [
        {
            "name": "focus_indie_semantic_filter",
            "query": "카페에서 집중할 때 듣는 차분한 노래",
            "where": {
                "$and": [
                    {"genre": {"$ne": "댄스"}},
                    {"lyrics_length": {"$gte": 20}},
                ]
            },
        },
        {
            "name": "rain_or_ballad_filter",
            "query": "비 오는 밤 감성 노래",
            "where": {
                "$or": [
                    {"primary_mood": {"$eq": "비"}},
                    {"genre": {"$eq": "발라드"}},
                ]
            },
        },
        {
            "name": "travel_genre_match",
            "query": "드라이브 갈 때 듣는 자유로운 노래",
            "where": {"genre": {"$eq": "인디"}},
        },
    ]


def build_pinecone_hybrid_scenarios() -> list[dict]:
    return [
        {
            "name": "focus_indie_semantic_filter",
            "query": "카페에서 집중할 때 듣는 차분한 노래",
            "filter": {
                "$and": [
                    {"genre": {"$ne": "댄스"}},
                    {"lyrics_length": {"$gte": 20}},
                ]
            },
        },
        {
            "name": "rain_or_ballad_filter",
            "query": "비 오는 밤 감성 노래",
            "filter": {
                "$or": [
                    {"primary_mood": {"$eq": "비"}},
                    {"genre": {"$eq": "발라드"}},
                ]
            },
        },
        {
            "name": "travel_genre_match",
            "query": "드라이브 갈 때 듣는 자유로운 노래",
            "filter": {
                "$and": [
                    {"genre": {"$eq": "인디"}},
                    {"mood_count": {"$gte": 1}},
                ]
            },
        },
    ]
