import unittest

from vector_db.benchmark import build_scaled_cache, flatten_benchmark_summary, summarize_benchmark_rows


class VectorBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.base_cache = {
            "songs": [
                {"id": "song_001"},
                {"id": "song_002"},
            ],
            "embeddings": [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
        }
        self.expanded_songs = [
            {"id": "song_001", "source_song_id": "song_001"},
            {"id": "song_002", "source_song_id": "song_002"},
            {"id": "song_001__variant_001", "source_song_id": "song_001"},
        ]

    def test_build_scaled_cache_reuses_source_song_embeddings_for_variants(self):
        scaled = build_scaled_cache(self.base_cache, self.expanded_songs)

        self.assertEqual(len(scaled["songs"]), 3)
        self.assertEqual(scaled["embeddings"][0], [0.1, 0.2])
        self.assertEqual(scaled["embeddings"][2], [0.1, 0.2])

    def test_summarize_benchmark_rows_groups_by_backend_and_size(self):
        summary = summarize_benchmark_rows(
            [
                {"backend": "file", "size": 100, "latency_ms": 5.0},
                {"backend": "file", "size": 100, "latency_ms": 7.0},
                {"backend": "chroma", "size": 100, "latency_ms": 4.0},
            ]
        )

        self.assertEqual(summary[("file", 100)]["runs"], 2)
        self.assertAlmostEqual(summary[("file", 100)]["avg_latency_ms"], 6.0)
        self.assertEqual(summary[("chroma", 100)]["runs"], 1)

    def test_flatten_benchmark_summary_returns_json_safe_rows(self):
        rows = flatten_benchmark_summary(
            {
                ("file", 100): {"runs": 2, "avg_latency_ms": 6.0, "min_latency_ms": 5.0, "max_latency_ms": 7.0},
            }
        )

        self.assertEqual(rows[0]["backend"], "file")
        self.assertEqual(rows[0]["size"], 100)
        self.assertEqual(rows[0]["runs"], 2)


if __name__ == "__main__":
    unittest.main()
