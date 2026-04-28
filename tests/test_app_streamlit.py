import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppStreamlitTests(unittest.TestCase):
    def test_results_page_omits_session_mix_summary_card(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        script = f"""
from pathlib import Path
source = Path(r"{app_path}").read_text(encoding="utf-8")
source = source.rsplit("\\nmain()", 1)[0]
namespace = {{}}
exec(source, namespace)
namespace["init_state"]()
namespace["st"].session_state["last_query"] = "카페 느낌"
namespace["st"].session_state["top_mood"] = "집중·공부"
namespace["st"].session_state["top_mood_score"] = 0.82
namespace["st"].session_state["result_summary"] = "sample summary"
namespace["st"].session_state["use_enriched"] = True
namespace["st"].session_state["selected_mood"] = "전체"
namespace["st"].session_state["comparison"] = {{"zero_overlap_semantic_results": []}}
namespace["st"].session_state["kw_results"] = []
namespace["st"].session_state["enriched_results"] = [{{
    "id": "song_001",
    "title": "군밤타령",
    "artist": "카더가든",
    "genre": "인디",
    "mood_tags": ["집중", "공부", "잔잔한"],
    "lyrics": "잔잔한 반복의 멜로디",
    "similarity": 0.91,
    "rank": 1,
    "youtube_music_url": "https://music.youtube.com/search?q=군밤타령"
}}]
namespace["st"].session_state["simple_results"] = namespace["st"].session_state["enriched_results"]
namespace["apply_theme"](namespace["current_theme"]())
namespace["render_results"]()
        """

        at = AppTest.from_string(script)

        at.run()

        self.assertFalse(any("Session Mix" in item.value for item in at.markdown))

    def test_map_page_omits_session_mix_summary_card(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        script = f"""
from pathlib import Path
import numpy as np
import tsne_visualizer
source = Path(r"{app_path}").read_text(encoding="utf-8")
source = source.rsplit("\\nmain()", 1)[0]
namespace = {{}}
exec(source, namespace)
namespace["init_state"]()
namespace["st"].session_state["last_query"] = "카페 느낌"
namespace["st"].session_state["top_mood"] = "집중·공부"
namespace["st"].session_state["top_mood_score"] = 0.82
namespace["st"].session_state["result_summary"] = "sample summary"
namespace["st"].session_state["use_enriched"] = True
namespace["st"].session_state["query_vec"] = np.array([1.0, 0.0])
namespace["st"].session_state["map_requested"] = True
namespace["st"].session_state["enriched_results"] = [{{
    "id": "song_001",
    "title": "군밤타령",
    "artist": "카더가든",
    "genre": "인디",
    "mood_tags": ["집중", "공부", "잔잔한"],
    "lyrics": "잔잔한 반복의 멜로디",
    "similarity": 0.91,
    "rank": 1,
    "youtube_music_url": "https://music.youtube.com/search?q=군밤타령"
}}]
namespace["st"].session_state["simple_results"] = namespace["st"].session_state["enriched_results"]
namespace["get_tsne_coords_for_query"] = lambda query_vec, enriched: (np.array([[0.0, 0.0]]), np.array([1.0, 1.0]))
tsne_visualizer._setup_korean_font = lambda: None
tsne_visualizer.load_embeddings = lambda enriched: (None, namespace["st"].session_state["enriched_results"])
namespace["apply_theme"](namespace["current_theme"]())
namespace["render_map"]()
        """

        at = AppTest.from_string(script)

        at.run()

        self.assertFalse(any("Session Mix" in item.value for item in at.markdown))
        self.assertTrue(
            any("각 점은 곡의 감정 임베딩을 2차원으로 축소한 위치입니다." in item.value for item in at.markdown)
        )
        self.assertTrue(
            any("별표는 현재 당신이 검색한 쿼리의 위치입니다." in item.value for item in at.markdown)
        )

    def test_results_page_omits_detail_selector_and_zero_overlap_expander(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        script = f"""
from pathlib import Path
source = Path(r"{app_path}").read_text(encoding="utf-8")
source = source.rsplit("\\nmain()", 1)[0]
namespace = {{}}
exec(source, namespace)
namespace["init_state"]()
namespace["st"].session_state["last_query"] = "카페 느낌"
namespace["st"].session_state["top_mood"] = "집중·공부"
namespace["st"].session_state["top_mood_score"] = 0.82
namespace["st"].session_state["result_summary"] = "sample summary"
namespace["st"].session_state["use_enriched"] = True
namespace["st"].session_state["selected_mood"] = "전체"
namespace["st"].session_state["comparison"] = {{"zero_overlap_semantic_results": []}}
namespace["st"].session_state["kw_results"] = []
namespace["st"].session_state["enriched_results"] = [{{
    "id": "song_001",
    "title": "군밤타령",
    "artist": "카더가든",
    "genre": "인디",
    "mood_tags": ["집중", "공부", "잔잔한"],
    "lyrics": "잔잔한 반복의 멜로디",
    "similarity": 0.91,
    "rank": 1,
    "youtube_music_url": "https://music.youtube.com/search?q=군밤타령"
}}]
namespace["st"].session_state["simple_results"] = namespace["st"].session_state["enriched_results"]
namespace["apply_theme"](namespace["current_theme"]())
namespace["render_results"]()
        """

        at = AppTest.from_string(script)

        at.run()

        self.assertEqual(len(at.selectbox), 0)
        self.assertEqual(len(at.expander), 0)

    def test_home_uses_fixed_top_8_without_result_count_slider(self):
        at = AppTest.from_file("app.py")

        at.run()

        self.assertEqual(at.session_state["top_k"], 8)
        self.assertEqual(at.session_state["last_search_settings"]["top_k"], 8)
        self.assertEqual(len(at.slider), 0)

    def test_chip_click_prefills_query_without_session_state_error(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        script = f"""
from pathlib import Path
source = Path(r"{app_path}").read_text(encoding="utf-8")
source = source.rsplit("\\nmain()", 1)[0]
namespace = {{}}
exec(source, namespace)
namespace["init_state"]()
namespace["apply_theme"](namespace["current_theme"]())
namespace["render_home"]()
        """

        at = AppTest.from_string(script)

        at.run()
        chip_button = next(
            button for button in at.button
            if button.key == "chip_드라이브 갈 때"
        )
        chip_button.click().run()

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["query_input"], "드라이브 갈 때")
        self.assertEqual(at.session_state["query_prefill"], "")
        self.assertEqual(at.session_state["pending_query"], "드라이브 갈 때")


if __name__ == "__main__":
    unittest.main()
