import json
import unittest
from pathlib import Path


class SongDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.songs = json.loads(Path("data/songs.json").read_text(encoding="utf-8"))

    def test_dataset_has_expanded_song_count(self):
        self.assertGreaterEqual(len(self.songs), 110)

    def test_dataset_has_multiple_cafe_focus_examples(self):
        terms = ["카페", "커피", "로파이", "노트북", "공부", "집중", "창가"]
        matches = []
        for song in self.songs:
            tags = set(song.get("mood_tags", []))
            text = " ".join(
                [
                    song.get("title", ""),
                    song.get("artist", ""),
                    song.get("lyrics", ""),
                    " ".join(song.get("mood_tags", [])),
                ]
            )
            if tags & {"집중", "공부"} and any(term in text for term in terms):
                matches.append(song["id"])

        self.assertGreaterEqual(len(matches), 8)

    def test_dataset_has_example_query_contexts(self):
        rainy_terms = ["비", "빗소리", "창가"]
        drive_terms = ["드라이브", "고속도로", "야경", "차창"]
        travel_terms = ["혼자", "여행", "열차", "바다"]

        rainy = 0
        drive = 0
        travel = 0

        for song in self.songs:
            tags = set(song.get("mood_tags", []))
            text = " ".join(
                [
                    song.get("title", ""),
                    song.get("artist", ""),
                    song.get("lyrics", ""),
                    " ".join(song.get("mood_tags", [])),
                ]
            )
            if "비" in tags and any(term in text for term in rainy_terms):
                rainy += 1
            if tags & {"여행", "자유"} and any(term in text for term in drive_terms):
                drive += 1
            if tags & {"여행", "자유"} and any(term in text for term in travel_terms):
                travel += 1

        self.assertGreaterEqual(rainy, 6)
        self.assertGreaterEqual(drive, 4)
        self.assertGreaterEqual(travel, 4)


if __name__ == "__main__":
    unittest.main()
