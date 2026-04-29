import unittest

from classify import rerank_mood_ranking


class ClassifyTests(unittest.TestCase):
    def test_rerank_mood_ranking_prefers_focus_for_generic_cafe_query(self):
        ranked = [
            ("신남·파티", 0.3263),
            ("집중·공부", 0.2369),
            ("새벽 감성", 0.2337),
            ("그리움·이별", 0.2124),
        ]

        reranked = rerank_mood_ranking("카페 분위기", ranked)

        self.assertEqual(reranked[0][0], "집중·공부")

    def test_rerank_mood_ranking_prefers_focus_for_cafe_queries(self):
        ranked = [
            ("그리움·이별", 0.1771),
            ("집중·공부", 0.1655),
            ("새벽 감성", 0.1644),
            ("신남·파티", 0.1611),
        ]

        reranked = rerank_mood_ranking("차분하지만 집중이 잘 되는 카페 분위기", ranked)

        self.assertEqual(reranked[0][0], "집중·공부")

    def test_rerank_mood_ranking_keeps_party_queries_lively(self):
        ranked = [
            ("집중·공부", 0.1820),
            ("신남·파티", 0.1760),
            ("여행·자유", 0.1700),
        ]

        reranked = rerank_mood_ranking("신나는 파티 음악", ranked)

        self.assertEqual(reranked[0][0], "신남·파티")


if __name__ == "__main__":
    unittest.main()
