import unittest
from pathlib import Path


class SearchResilienceTests(unittest.TestCase):
    def _load_namespace(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        source = app_path.read_text(encoding="utf-8")
        source = source.rsplit("\nmain()", 1)[0]
        namespace = {}
        exec(source, namespace)
        namespace["init_state"]()
        return namespace

    def test_execute_search_falls_back_to_keyword_results_when_embedding_fails(self):
        namespace = self._load_namespace()
        sample_results = [
            {
                "id": "song_001",
                "title": "Fallback Song",
                "artist": "MoodTune",
                "genre": "indie",
                "lyrics": "rain window cafe focus",
                "mood_tags": ["focus"],
                "youtube_music_url": "https://music.youtube.com/watch?v=1",
                "rank": 1,
                "tfidf_similarity": 0.91,
                "common_keywords": ["rain"],
                "keyword_overlap_count": 1,
            }
        ]

        def raise_connection_error(query):
            raise RuntimeError("connection error")

        namespace["embed_text"] = raise_connection_error
        namespace["keyword_search"] = lambda query, top_k=5: [dict(row) for row in sample_results]
        namespace["compare_search_results"] = lambda semantic_results, keyword_results, query: {
            "query": query,
            "overlap_count": 0,
            "semantic_only": [],
            "keyword_only": [row["id"] for row in keyword_results],
            "zero_overlap_semantic_results": [],
        }

        result = namespace["execute_search"]("rainy cafe")

        self.assertEqual(result["simple_results"], sample_results)
        self.assertEqual(result["enriched_results"], sample_results)
        self.assertEqual(result["top_mood"], namespace["DEFAULT_MOOD"])
        self.assertEqual(result["top_mood_score"], 0.0)
        self.assertIsNone(result["query_vec"])
        self.assertIn("keyword", result["result_summary"].lower())
        self.assertIn("connection error", result["result_summary"].lower())
        self.assertEqual(result["comparison"]["keyword_only"], ["song_001"])


if __name__ == "__main__":
    unittest.main()
