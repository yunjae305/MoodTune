import unittest

import numpy as np

from vector_db.visualizer import build_color_labels, reduce_embeddings


class VectorVisualizerTests(unittest.TestCase):
    def test_build_color_labels_uses_requested_metadata_field(self):
        labels = build_color_labels(
            [
                {"genre": "인디", "primary_mood": "비"},
                {"genre": "발라드", "primary_mood": "집중"},
            ],
            color_by="genre",
        )

        self.assertEqual(labels, ["인디", "발라드"])

    def test_reduce_embeddings_returns_2d_projection(self):
        embeddings = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.1, 0.4],
                [0.9, 0.8, 0.7],
                [0.8, 0.9, 0.6],
                [0.5, 0.4, 0.6],
            ]
        )

        coords = reduce_embeddings(embeddings, method="tsne", perplexity=2)

        self.assertEqual(coords.shape, (5, 2))


if __name__ == "__main__":
    unittest.main()
