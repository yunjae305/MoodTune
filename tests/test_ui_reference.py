import unittest

from ui_reference import (
    QUERY_CHIPS,
    build_search_state_update,
    build_compare_rows,
    build_result_reason,
    build_result_summary,
    consume_query_prefill,
    get_mood_theme,
    get_sidebar_moods,
    truncate_text,
)


class UiReferenceTests(unittest.TestCase):
    def test_query_chips_use_mood_or_situation_style_examples(self):
        self.assertEqual(
            QUERY_CHIPS[:4],
            [
                "드라이브 갈 때",
                "비 오는 창가에서",
                "공부할 때 듣는 노래",
                "카페 분위기",
            ],
        )

    def test_get_mood_theme_returns_zip_style_theme(self):
        theme = get_mood_theme("비 오는 날")

        self.assertEqual(theme["id"], "rain")
        self.assertEqual(theme["label"], "비 오는 날")
        self.assertEqual(theme["accent"], "#42a5f5")

    def test_get_sidebar_moods_follows_defined_order(self):
        moods = get_sidebar_moods()

        self.assertEqual(moods[0]["category"], "새벽 감성")
        self.assertEqual(moods[-1]["category"], "여행·자유")
        self.assertEqual(len(moods), 8)

    def test_truncate_text_collapses_whitespace_and_limits_length(self):
        text = "비 오는   날\n혼자 듣기 좋은 노래가 필요해"

        self.assertEqual(truncate_text(text, limit=15), "비 오는 날 혼자 듣기…")

    def test_build_result_reason_uses_song_tags_and_detected_mood(self):
        reason = build_result_reason(
            {"genre": "발라드", "mood_tags": ["비", "감성적"]},
            "비 오는 날",
        )

        self.assertIn("비, 감성적", reason)
        self.assertIn("비 오는 날", reason)

    def test_build_result_summary_mentions_top_song_when_present(self):
        summary = build_result_summary(
            "비 오는 날 혼자 듣기 좋은 노래",
            "비 오는 날",
            0.89,
            {"title": "비도 오고 그래서", "artist": "멜로망스"},
        )

        self.assertIn("0.89", summary)
        self.assertIn("비도 오고 그래서", summary)

    def test_build_compare_rows_aligns_simple_and_enriched_results_by_rank(self):
        rows = build_compare_rows(
            [{"title": "A"}, {"title": "B"}],
            [{"title": "X"}],
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["simple"]["title"], "A")
        self.assertEqual(rows[0]["enriched"]["title"], "X")
        self.assertIsNone(rows[1]["enriched"])

    def test_build_search_state_update_uses_prefill_for_preset_sources(self):
        update = build_search_state_update("  비 오는 날 혼자 듣기 좋은 노래  ", source="chip")

        self.assertEqual(update["pending_query"], "비 오는 날 혼자 듣기 좋은 노래")
        self.assertEqual(update["query_prefill"], "비 오는 날 혼자 듣기 좋은 노래")
        self.assertEqual(update["view"], "matching")

    def test_consume_query_prefill_returns_widget_safe_updates(self):
        updates = consume_query_prefill({"query_prefill": "봄날 설렘"})

        self.assertEqual(
            updates,
            {"query_input": "봄날 설렘", "query_prefill": ""},
        )


if __name__ == "__main__":
    unittest.main()
