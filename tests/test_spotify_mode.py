import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class SpotifyModeTests(unittest.TestCase):
    def _load_namespace(self) -> dict:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        source = app_path.read_text(encoding="utf-8")
        source = source.rsplit("\nmain()", 1)[0]
        namespace = {}
        exec(source, namespace)
        namespace["init_state"]()
        return namespace

    def test_execute_search_in_spotify_mode_skips_embedding_pipeline(self):
        namespace = self._load_namespace()
        namespace["st"].session_state["use_spotify"] = True
        namespace["st"].session_state["music_region"] = "국내"

        spotify_results = [
            {
                "id": "spotify_001",
                "title": "Levitating",
                "artist": "Dua Lipa",
                "genre": "pop",
                "lyrics": "",
                "youtube_music_url": "https://music.youtube.com/search?q=Levitating",
                "source": "spotify",
            }
        ]

        class FakeSpotifyClient:
            def search_recommendations(self, features, limit, region, query):
                return spotify_results

        namespace["SpotifyClient"] = FakeSpotifyClient
        namespace["map_mood_to_spotify_features"] = lambda query: {"target_energy": 0.8}
        namespace["embed_text"] = lambda query: (_ for _ in ()).throw(
            AssertionError("Spotify 모드에서는 임베딩을 호출하면 안 됩니다.")
        )
        namespace["generate_cached_result_summary"] = lambda **kwargs: "Spotify 요약"

        result = namespace["execute_search"]("신나는 드라이브 음악")

        self.assertEqual(result["enriched_results"], spotify_results)
        self.assertEqual(result["simple_results"], spotify_results)
        self.assertEqual(result["mood_ranking"], [])
        self.assertEqual(result["top_mood"], "Spotify 추천")
        self.assertEqual(result["top_mood_score"], 0.0)
        self.assertIsNone(result["query_vec"])
        self.assertEqual(result["result_summary"], "Spotify 요약")
        self.assertIsNone(result["comparison"])

    def test_execute_search_in_spotify_mode_does_not_fallback_to_embeddings_when_no_result(self):
        namespace = self._load_namespace()
        namespace["st"].session_state["use_spotify"] = True
        namespace["st"].session_state["music_region"] = "국내"

        class FakeSpotifyClient:
            def search_recommendations(self, features, limit, region, query):
                return []

        namespace["SpotifyClient"] = FakeSpotifyClient
        namespace["map_mood_to_spotify_features"] = lambda query: {"target_valence": 0.4}
        namespace["embed_text"] = lambda query: (_ for _ in ()).throw(
            AssertionError("Spotify 모드에서는 결과가 없어도 임베딩으로 폴백하면 안 됩니다.")
        )

        result = namespace["execute_search"]("조용한 야외 밤")

        self.assertEqual(result["enriched_results"], [])
        self.assertEqual(result["simple_results"], [])
        self.assertEqual(result["mood_ranking"], [])
        self.assertEqual(result["top_mood"], "Spotify 추천")
        self.assertEqual(result["top_mood_score"], 0.0)
        self.assertIsNone(result["query_vec"])
        self.assertIsNone(result["comparison"])

    def test_home_turns_off_enriched_embedding_when_spotify_mode_enabled(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        script = f"""
from pathlib import Path
source = Path(r"{app_path}").read_text(encoding="utf-8")
source = source.rsplit("\\nmain()", 1)[0]
namespace = {{}}
exec(source, namespace)
namespace["init_state"]()
namespace["st"].session_state["use_spotify"] = True
namespace["st"].session_state["use_enriched"] = True
namespace["apply_theme"](namespace["current_theme"]())
namespace["render_home"]()
        """

        at = AppTest.from_string(script)

        at.run(timeout=10)

        self.assertFalse(at.session_state["use_enriched"])
        self.assertTrue(
            any("Spotify 추천이 켜져 있으면 임베딩 검색과 무드 분류는 비활성화됩니다." in item.value for item in at.markdown)
        )

    def test_home_does_not_show_spotify_category_options(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        script = f"""
from pathlib import Path
source = Path(r"{app_path}").read_text(encoding="utf-8")
source = source.rsplit("\\nmain()", 1)[0]
namespace = {{}}
exec(source, namespace)
namespace["init_state"]()
namespace["st"].session_state["use_spotify"] = True
namespace["apply_theme"](namespace["current_theme"]())
namespace["render_home"]()
        """

        at = AppTest.from_string(script)

        at.run(timeout=10)

        self.assertFalse(
            any(getattr(item, "options", None) == ["전체", "J-POP", "게임 OST"] for item in at.radio)
        )

    def test_render_map_in_spotify_mode_blocks_embedding_view(self):
        namespace = self._load_namespace()
        namespace["st"].session_state["use_spotify"] = True
        captured = {}

        def fake_render_empty_state(title, description):
            captured["title"] = title
            captured["description"] = description

        namespace["render_empty_state"] = fake_render_empty_state

        namespace["render_map"]()

        self.assertIn("Spotify", captured["description"])


if __name__ == "__main__":
    unittest.main()
