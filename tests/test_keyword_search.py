import unittest

from keyword_search import compare_search_results


class CompareSearchResultsTests(unittest.TestCase):
    def test_compare_search_results_detects_zero_overlap_semantic_results(self):
        semantic_results = [
            {"id": "song_001", "lyrics": "따뜻한 위로가 필요해"},
            {"id": "song_002", "lyrics": "달빛 아래 혼자 걸어"},
        ]
        keyword_results = [
            {"id": "song_002"},
            {"id": "song_003"},
        ]

        comparison = compare_search_results(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            query="달빛 산책",
        )

        self.assertEqual(comparison["overlap_count"], 1)
        self.assertEqual(comparison["semantic_only"], ["song_001"])
        self.assertEqual(comparison["keyword_only"], ["song_003"])
        self.assertEqual(
            [song["id"] for song in comparison["zero_overlap_semantic_results"]],
            ["song_001"],
        )


if __name__ == "__main__":
    unittest.main()
