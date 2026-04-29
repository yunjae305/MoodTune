import os
import unittest

from ai_summary import DEFAULT_SUMMARY_MODEL, generate_result_summary


class _FakeResponse:
    def __init__(self, text: str):
        self.output_text = text


class _FakeResponses:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.text)


class _FakeClient:
    def __init__(self, text: str):
        self.responses = _FakeResponses(text)


class _FailingResponses:
    def create(self, **kwargs):
        raise RuntimeError("api error")


class _FailingClient:
    def __init__(self):
        self.responses = _FailingResponses()


class AiSummaryTests(unittest.TestCase):
    def test_default_summary_model_matches_assignment_requirement(self):
        self.assertEqual(DEFAULT_SUMMARY_MODEL, "gpt-4.1-mini")

    def test_generate_result_summary_uses_gpt_response_text_when_available(self):
        client = _FakeClient("지금 기분에는 잔잔하고 집중감 있는 곡들이 잘 맞습니다.")

        summary = generate_result_summary(
            query="카페에서 집중하고 싶어",
            top_mood="집중·공부",
            top_score=0.91,
            results=[
                {"title": "군밤타령", "artist": "카더가든", "mood_tags": ["집중", "공부", "잔잔한"]},
                {"title": "있어줘", "artist": "죠지", "mood_tags": ["집중", "공부", "잔잔한"]},
            ],
            client=client,
        )

        self.assertEqual(summary, "지금 기분에는 잔잔하고 집중감 있는 곡들이 잘 맞습니다.")
        self.assertEqual(client.responses.calls[0]["model"], "gpt-4.1-mini")

    def test_generate_result_summary_uses_model_from_environment_when_present(self):
        previous = os.environ.get("OPENAI_SUMMARY_MODEL")
        os.environ["OPENAI_SUMMARY_MODEL"] = "gpt-4.1-mini"
        client = _FakeClient("환경변수 모델이 적용된 요약입니다.")

        try:
            summary = generate_result_summary(
                query="비 오는 날 혼자 듣기 좋은 노래",
                top_mood="비 오는 날",
                top_score=0.89,
                results=[
                    {"title": "비도 오고 그래서", "artist": "멜로망스", "mood_tags": ["비", "감성적"]},
                ],
                client=client,
            )
        finally:
            if previous is None:
                os.environ.pop("OPENAI_SUMMARY_MODEL", None)
            else:
                os.environ["OPENAI_SUMMARY_MODEL"] = previous

        self.assertEqual(summary, "환경변수 모델이 적용된 요약입니다.")
        self.assertEqual(client.responses.calls[0]["model"], "gpt-4.1-mini")

    def test_generate_result_summary_falls_back_when_api_call_fails(self):
        summary = generate_result_summary(
            query="비 오는 날 혼자 듣기 좋은 노래",
            top_mood="비 오는 날",
            top_score=0.89,
            results=[
                {"title": "비도 오고 그래서", "artist": "멜로망스", "mood_tags": ["비", "감성적"]},
            ],
            client=_FailingClient(),
        )

        self.assertIn("비 오는 날", summary)
        self.assertIn("비도 오고 그래서", summary)


if __name__ == "__main__":
    unittest.main()
