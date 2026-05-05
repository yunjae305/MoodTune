import unittest

from vector_db.data import (
    build_song_document,
    build_song_metadata,
    build_vector_records,
    expand_song_dataset,
)


class VectorDataTests(unittest.TestCase):
    def setUp(self):
        self.song = {
            "id": "song_001",
            "title": "Rainy Focus",
            "artist": "MoodTune",
            "genre": "인디",
            "lyrics": "조용한 카페 창가에 비가 내리고 집중이 잘 되는 밤",
            "mood_tags": ["비", "집중"],
            "youtube_music_url": "https://music.youtube.com/watch?v=1",
        }

    def test_build_song_document_combines_song_fields(self):
        document = build_song_document(self.song)

        self.assertIn("Rainy Focus", document)
        self.assertIn("MoodTune", document)
        self.assertIn("인디", document)
        self.assertIn("조용한 카페", document)

    def test_build_song_metadata_adds_filterable_fields(self):
        metadata = build_song_metadata(
            self.song,
            tenant="main",
            dataset_name="base",
            variant_index=0,
        )

        self.assertEqual(metadata["song_id"], "song_001")
        self.assertEqual(metadata["genre"], "인디")
        self.assertEqual(metadata["tenant"], "main")
        self.assertEqual(metadata["dataset_name"], "base")
        self.assertEqual(metadata["variant_index"], 0)
        self.assertGreaterEqual(metadata["lyrics_length"], 10)
        self.assertEqual(metadata["mood_count"], 2)
        self.assertTrue(metadata["genre_key"].startswith("genre_"))

    def test_build_vector_records_keeps_id_document_and_metadata_together(self):
        records = build_vector_records([self.song], tenant="main", dataset_name="base")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "song_001")
        self.assertIn("Rainy Focus", records[0]["document"])
        self.assertEqual(records[0]["metadata"]["song_id"], "song_001")

    def test_expand_song_dataset_reaches_target_size_with_unique_ids(self):
        songs = [self.song, {**self.song, "id": "song_002", "title": "Night Drive"}]

        expanded = expand_song_dataset(songs, target_size=5)

        self.assertEqual(len(expanded), 5)
        self.assertEqual(len({song["id"] for song in expanded}), 5)
        self.assertEqual(expanded[0]["id"], "song_001")
        self.assertEqual(expanded[1]["id"], "song_002")
        self.assertTrue(any(song["id"].startswith("song_001__variant_") for song in expanded))


if __name__ == "__main__":
    unittest.main()
