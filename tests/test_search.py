import unittest

import numpy as np

from search import prioritize_indices_by_mood, search_from_query_vector


class SearchRankingTests(unittest.TestCase):
    def test_prioritize_indices_by_mood_moves_matching_songs_first(self):
        songs = [
            {"id": "song_001", "mood_tags": ["위로"]},
            {"id": "song_002", "mood_tags": ["비"]},
            {"id": "song_003", "mood_tags": ["비"]},
        ]
        similarities = np.array([0.99, 0.70, 0.80])

        indices = prioritize_indices_by_mood(
            songs=songs,
            similarities=similarities,
            top_k=3,
            top_mood="비 오는 날",
        )

        self.assertEqual(indices, [2, 1, 0])

    def test_search_from_query_vector_prioritizes_detected_mood_before_fallback(self):
        cache = {
            "songs": [
                {"id": "song_001", "title": "A", "artist": "AA", "mood_tags": ["위로"]},
                {"id": "song_002", "title": "B", "artist": "BB", "mood_tags": ["비"]},
                {"id": "song_003", "title": "C", "artist": "CC", "mood_tags": ["비"]},
            ],
            "embeddings": [
                [1.0, 0.0],
                [0.7, 0.7],
                [0.8, 0.6],
            ],
        }

        results = search_from_query_vector(
            query_vec=np.array([1.0, 0.0]),
            cache=cache,
            top_k=3,
            prioritized_mood="비 오는 날",
        )

        self.assertEqual([song["id"] for song in results], ["song_003", "song_002", "song_001"])
        self.assertEqual([song["rank"] for song in results], [1, 2, 3])

    def test_search_from_query_vector_matches_category_names_to_short_song_tags(self):
        cache = {
            "songs": [
                {"id": "song_001", "title": "A", "artist": "AA", "mood_tags": ["비"]},
                {"id": "song_002", "title": "B", "artist": "BB", "mood_tags": ["새벽"]},
                {"id": "song_003", "title": "C", "artist": "CC", "mood_tags": ["위로"]},
            ],
            "embeddings": [
                [0.8, 0.6],
                [1.0, 0.0],
                [0.7, 0.7],
            ],
        }

        results = search_from_query_vector(
            query_vec=np.array([1.0, 0.0]),
            cache=cache,
            top_k=3,
            prioritized_mood="비 오는 날",
        )

        self.assertEqual(results[0]["id"], "song_001")


if __name__ == "__main__":
    unittest.main()
