import unittest

from spotify_mapper import keyword_override_features, sanitize_spotify_features


class SpotifyMapperTests(unittest.TestCase):
    def test_sanitize_spotify_features_normalizes_lists_and_ranges(self):
        payload = sanitize_spotify_features(
            {
                "seed_genres": "ambient, chill, ambient",
                "search_terms": "dream pop, rainy night",
                "target_energy": 1.4,
                "target_valence": -0.2,
                "target_danceability": "0.6",
                "target_tempo": 220,
            }
        )

        self.assertEqual(payload["seed_genres"], ["ambient", "chill"])
        self.assertEqual(payload["search_terms"], ["dream pop", "rainy night"])
        self.assertEqual(payload["target_energy"], 1.0)
        self.assertEqual(payload["target_valence"], 0.0)
        self.assertEqual(payload["target_danceability"], 0.6)
        self.assertEqual(payload["target_tempo"], 180)

    def test_sanitize_spotify_features_falls_back_to_defaults(self):
        payload = sanitize_spotify_features({})

        self.assertEqual(payload["seed_genres"], ["pop"])
        self.assertEqual(payload["search_terms"], ["pop"])
        self.assertEqual(payload["target_energy"], 0.5)
        self.assertEqual(payload["target_valence"], 0.5)
        self.assertEqual(payload["target_danceability"], 0.5)
        self.assertEqual(payload["target_tempo"], 120)

    def test_keyword_override_features_boosts_workout_queries(self):
        payload = keyword_override_features("운동할 때 들을 강한 에너지의 음악")

        self.assertIn("work-out", payload["seed_genres"])
        self.assertIn("edm", payload["seed_genres"])
        self.assertIn("workout", payload["search_terms"])
        self.assertGreaterEqual(payload["target_energy"], 0.8)
        self.assertGreaterEqual(payload["target_danceability"], 0.7)


if __name__ == "__main__":
    unittest.main()
