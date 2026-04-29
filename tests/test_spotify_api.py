import unittest

from spotify_api import SpotifyClient, build_search_queries


class FakeSpotifyApi:
    def __init__(self):
        self.calls = []

    def search(self, q, type, limit, market):
        self.calls.append((q, market, limit))
        items = []
        if "새벽 감성" in q:
            items.append(self._track("track_1", "Moonlight", 88))
            items.append(self._track("track_2", "Night Drive", 80))
        if "ambient" in q or "chill" in q:
            items.append(self._track("track_1", "Moonlight", 88))
            items.append(self._track("track_3", "Blue Haze", 76))
        if "calm" in q or "dreamy" in q:
            items.append(self._track("track_4", "Dream Loop", 74))
        return {"tracks": {"items": items[:limit]}}

    def _track(self, track_id, name, popularity):
        return {
            "id": track_id,
            "name": name,
            "artists": [{"name": "Test Artist"}],
            "album": {"name": "Test Album", "images": []},
            "external_urls": {"spotify": f"https://open.spotify.com/track/{track_id}"},
            "preview_url": None,
            "popularity": popularity,
        }


class SpotifyApiTests(unittest.TestCase):
    def test_build_search_queries_reflects_query_features_and_region(self):
        calm_queries = build_search_queries(
            query="새벽 감성의 몽환적인 음악",
            seed_genres=["ambient", "chill"],
            region="국외",
            features={
                "search_terms": ["dream pop", "late night"],
                "target_energy": 0.2,
                "target_valence": 0.4,
                "target_danceability": 0.2,
            },
        )
        workout_queries = build_search_queries(
            query="운동할 때 듣는 강한 음악",
            seed_genres=["edm", "work-out"],
            region="국내",
            features={
                "target_energy": 0.9,
                "target_valence": 0.7,
                "target_danceability": 0.8,
            },
        )

        self.assertNotEqual(calm_queries, workout_queries)
        self.assertTrue(any("ambient" in item.lower() or "chill" in item.lower() for item in calm_queries))
        self.assertTrue(any("dream pop" in item.lower() or "late night" in item.lower() for item in calm_queries))
        self.assertTrue(any("k-pop" in item.lower() or "korean" in item.lower() for item in workout_queries))

    def test_search_recommendations_combines_multiple_query_variants(self):
        client = SpotifyClient.__new__(SpotifyClient)
        client.client_id = "id"
        client.client_secret = "secret"
        client.user_token = None
        fake_sp = FakeSpotifyApi()
        client.sp = fake_sp

        results = client.search_recommendations(
            {
                "seed_genres": ["ambient", "chill"],
                "target_energy": 0.2,
                "target_valence": 0.4,
                "target_danceability": 0.2,
            },
            limit=3,
            region="국외",
            query="새벽 감성",
        )

        self.assertGreaterEqual(len(fake_sp.calls), 2)
        self.assertEqual([song["id"] for song in results], ["track_1", "track_2", "track_3"])
        self.assertTrue(all(song["source"] == "spotify" for song in results))
        self.assertTrue(all(song["youtube_music_url"].startswith("https://open.spotify.com/track/") for song in results))

    def test_build_search_queries_adds_english_hints_for_korean_mood_query(self):
        queries = build_search_queries(
            query="비 오는 밤에 듣고 싶은 잔잔한 노래",
            seed_genres=["ambient", "indie-pop"],
            region="국외",
            features={
                "target_energy": 0.2,
                "target_valence": 0.3,
                "target_danceability": 0.2,
                "target_tempo": 78,
            },
        )

        self.assertTrue(any("rainy" in item.lower() or "night" in item.lower() for item in queries))


if __name__ == "__main__":
    unittest.main()
