import unittest
from pathlib import Path
from unittest.mock import patch

from vector_db.services import run_benchmark


class FakeEmbedder:
    def embed_text(self, text):
        return [0.1, 0.2]


class FakeStore:
    def __init__(self, fail_once=False):
        self.fail_once = fail_once
        self.reset_calls = []
        self.sync_calls = []
        self.query_calls = 0

    def reset_scopes(self, scopes):
        self.reset_calls.append(list(scopes))

    def sync_records(self, records, skip_existing=True):
        self.sync_calls.append(
            {
                "ids": [record["id"] for record in records],
                "skip_existing": skip_existing,
            }
        )
        return {"all": len(records)}

    def query(self, query, top_k=5):
        self.query_calls += 1
        if self.fail_once and self.query_calls == 1:
            raise RuntimeError("Error creating hnsw segment reader: Nothing found on disk")
        return []


class VectorServicesTests(unittest.TestCase):
    def setUp(self):
        self.songs = [
            {
                "id": "song_001",
                "title": "Rain",
                "artist": "MoodTune",
                "genre": "indie",
                "mood_tags": ["calm"],
                "lyrics": "rain rain",
            }
        ]
        self.records = [
            {
                "id": "song_001",
                "document": "title: Rain",
                "metadata": {
                    "genre_key": "genre_a",
                },
            }
        ]

    def test_run_benchmark_resets_vector_stores_before_syncing(self):
        chroma_store = FakeStore()
        pinecone_store = FakeStore()

        with (
            patch("vector_db.services.load_base_cache", return_value={"songs": self.songs, "embeddings": [[0.1, 0.2]]}),
            patch("vector_db.services.load_song_dataset", return_value=self.songs),
            patch("vector_db.services.build_benchmark_datasets", return_value={100: self.songs}),
            patch("vector_db.services.build_vector_records", return_value=self.records),
            patch("vector_db.services.build_scaled_cache", return_value={"songs": self.songs, "embeddings": [[0.1, 0.2]]}),
            patch("vector_db.services.create_embedder", return_value=FakeEmbedder()),
            patch("vector_db.services.benchmark_file_search", return_value=[1.0]),
            patch("vector_db.services.create_chroma_store", return_value=chroma_store),
            patch("vector_db.services.create_pinecone_store", return_value=pinecone_store),
            patch("vector_db.services.plot_benchmark_summary", return_value=Path("cache/vector_benchmark.png")),
        ):
            run_benchmark("rain", sizes=(100,), runs=1)

        self.assertEqual(chroma_store.reset_calls, [["all", "genre_a"]])
        self.assertEqual(pinecone_store.reset_calls, [["all", "genre_a"]])
        self.assertEqual(chroma_store.sync_calls[0]["skip_existing"], False)
        self.assertEqual(pinecone_store.sync_calls[0]["skip_existing"], False)

    def test_run_benchmark_repairs_chroma_missing_segment_error_once(self):
        chroma_store = FakeStore(fail_once=True)
        pinecone_store = FakeStore()

        with (
            patch("vector_db.services.load_base_cache", return_value={"songs": self.songs, "embeddings": [[0.1, 0.2]]}),
            patch("vector_db.services.load_song_dataset", return_value=self.songs),
            patch("vector_db.services.build_benchmark_datasets", return_value={100: self.songs}),
            patch("vector_db.services.build_vector_records", return_value=self.records),
            patch("vector_db.services.build_scaled_cache", return_value={"songs": self.songs, "embeddings": [[0.1, 0.2]]}),
            patch("vector_db.services.create_embedder", return_value=FakeEmbedder()),
            patch("vector_db.services.benchmark_file_search", return_value=[1.0]),
            patch("vector_db.services.create_chroma_store", return_value=chroma_store),
            patch("vector_db.services.create_pinecone_store", return_value=pinecone_store),
            patch("vector_db.services.plot_benchmark_summary", return_value=Path("cache/vector_benchmark.png")),
        ):
            rows, _, _ = run_benchmark("rain", sizes=(100,), runs=1)

        self.assertEqual(len(rows), 3)
        self.assertEqual(len(chroma_store.reset_calls), 2)
        self.assertEqual(len(chroma_store.sync_calls), 2)


if __name__ == "__main__":
    unittest.main()
