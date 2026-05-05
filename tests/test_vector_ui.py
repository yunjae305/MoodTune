import unittest

from vector_db.ui import build_backend_filter, build_vector_result_markup


class VectorUiTests(unittest.TestCase):
    def test_build_backend_filter_combines_selected_controls(self):
        filter_expr = build_backend_filter(
            backend="chroma",
            genre="인디",
            primary_mood="비",
            min_lyrics_length=120,
            exclude_genre="댄스",
        )

        self.assertIn("$and", filter_expr)
        self.assertIn("genre", str(filter_expr))
        self.assertIn("$gte", str(filter_expr))
        self.assertIn("$ne", str(filter_expr))

    def test_build_vector_result_markup_includes_score_and_metadata(self):
        markup = build_vector_result_markup(
            {
                "id": "song_001",
                "title": "Rainy Focus",
                "artist": "MoodTune",
                "genre": "인디",
                "primary_mood": "비",
                "score": 0.88,
                "distance": 0.12,
            }
        )

        self.assertIn("Rainy Focus", markup)
        self.assertIn("0.8800", markup)
        self.assertIn("인디", markup)
        self.assertIn("비", markup)


if __name__ == "__main__":
    unittest.main()
