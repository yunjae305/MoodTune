import unittest

from vector_db.workflows import (
    build_benchmark_datasets,
    build_chroma_hybrid_scenarios,
    build_pinecone_hybrid_scenarios,
)


class VectorWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.songs = [
            {
                "id": f"song_{idx:03d}",
                "title": f"Song {idx}",
                "artist": "MoodTune",
                "genre": "발라드" if idx % 2 else "인디",
                "lyrics": "비 오는 밤 카페 집중 플레이리스트",
                "mood_tags": ["비", "집중"],
                "youtube_music_url": "https://music.youtube.com/watch?v=1",
            }
            for idx in range(1, 11)
        ]

    def test_build_benchmark_datasets_matches_requested_scales(self):
        datasets = build_benchmark_datasets(self.songs, scales=(5, 12, 20))

        self.assertEqual(sorted(datasets.keys()), [5, 12, 20])
        self.assertEqual(len(datasets[5]), 5)
        self.assertEqual(len(datasets[12]), 12)
        self.assertEqual(len(datasets[20]), 20)

    def test_build_chroma_hybrid_scenarios_contains_required_operators(self):
        scenarios = build_chroma_hybrid_scenarios()

        self.assertEqual(len(scenarios), 3)
        filters = [scenario["where"] for scenario in scenarios]
        self.assertTrue(any("genre" in where for where in filters))
        self.assertTrue(any("$and" in where for where in filters))
        self.assertTrue(any("$or" in where for where in filters))
        self.assertTrue(any("$gte" in str(where) for where in filters))
        self.assertTrue(any("$ne" in str(where) for where in filters))

    def test_build_pinecone_hybrid_scenarios_contains_equivalent_filters(self):
        scenarios = build_pinecone_hybrid_scenarios()

        self.assertEqual(len(scenarios), 3)
        filters = [scenario["filter"] for scenario in scenarios]
        self.assertTrue(any("$and" in filter_expr for filter_expr in filters))
        self.assertTrue(any("$or" in filter_expr for filter_expr in filters))
        self.assertTrue(any("$eq" in str(filter_expr) for filter_expr in filters))
        self.assertTrue(any("$gte" in str(filter_expr) for filter_expr in filters))


if __name__ == "__main__":
    unittest.main()
