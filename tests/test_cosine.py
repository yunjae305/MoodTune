import unittest

import numpy as np

from cosine import cosine_similarity, cosine_similarity_batch, verify_against_scipy


class CosineTests(unittest.TestCase):
    def test_cosine_similarity_returns_expected_values_for_basic_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        d = np.array([-1.0, 0.0, 0.0])

        self.assertAlmostEqual(cosine_similarity(a, b), 1.0, places=7)
        self.assertAlmostEqual(cosine_similarity(a, c), 0.0, places=7)
        self.assertAlmostEqual(cosine_similarity(a, d), -1.0, places=7)

    def test_cosine_similarity_batch_matches_scalar_results(self):
        query = np.array([1.0, 2.0, 3.0])
        matrix = np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [-1.0, -2.0, -3.0],
            ]
        )

        batch_scores = cosine_similarity_batch(query, matrix)
        scalar_scores = np.array([cosine_similarity(query, row) for row in matrix])

        np.testing.assert_allclose(batch_scores, scalar_scores, atol=1e-9)

    def test_verify_against_scipy_reports_valid_result(self):
        rng = np.random.default_rng(42)
        vec_a = rng.standard_normal(1536)
        vec_b = rng.standard_normal(1536)

        result = verify_against_scipy(vec_a, vec_b)

        self.assertTrue(result["is_valid"])
        self.assertLess(result["error"], result["tolerance"])


if __name__ == "__main__":
    unittest.main()
