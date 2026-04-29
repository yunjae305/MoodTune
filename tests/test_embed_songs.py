import os
import tempfile
import unittest
from pathlib import Path

from embed_songs import get_api_key, is_cache_valid


class EmbedSongsEnvTests(unittest.TestCase):
    def test_get_api_key_loads_value_from_dotenv_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")

            previous = os.environ.pop("OPENAI_API_KEY", None)
            try:
                self.assertEqual(get_api_key(env_path=env_path), "test-key")
            finally:
                if previous is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = previous

    def test_get_api_key_raises_clear_error_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"

            previous = os.environ.pop("OPENAI_API_KEY", None)
            try:
                with self.assertRaises(ValueError) as context:
                    get_api_key(env_path=env_path)
            finally:
                if previous is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = previous

        self.assertIn("OPENAI_API_KEY", str(context.exception))


class EmbedSongsCacheTests(unittest.TestCase):
    def test_is_cache_valid_returns_false_when_cache_missing(self):
        songs = [{"id": "song_001"}, {"id": "song_002"}]

        self.assertFalse(is_cache_valid(None, songs))

    def test_is_cache_valid_returns_false_when_song_ids_change(self):
        songs = [{"id": "song_001"}, {"id": "song_002"}]
        cache = {
            "songs": [{"id": "song_001"}, {"id": "song_003"}],
            "embeddings": [[0.1], [0.2]],
        }

        self.assertFalse(is_cache_valid(cache, songs))

    def test_is_cache_valid_returns_true_for_matching_song_ids(self):
        songs = [{"id": "song_001"}, {"id": "song_002"}]
        cache = {
            "songs": [{"id": "song_001"}, {"id": "song_002"}],
            "embeddings": [[0.1], [0.2]],
        }

        self.assertTrue(is_cache_valid(cache, songs))

    def test_is_cache_valid_returns_false_when_song_content_changes(self):
        songs = [
            {"id": "song_001", "lyrics": "new lyrics"},
            {"id": "song_002", "lyrics": "same lyrics"},
        ]
        cache = {
            "songs": [
                {"id": "song_001", "lyrics": "old lyrics"},
                {"id": "song_002", "lyrics": "same lyrics"},
            ],
            "embeddings": [[0.1], [0.2]],
        }

        self.assertFalse(is_cache_valid(cache, songs))


if __name__ == "__main__":
    unittest.main()
