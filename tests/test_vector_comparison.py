import unittest

from vector_db.comparison import build_backend_comparison_rows


class VectorComparisonTests(unittest.TestCase):
    def test_build_backend_comparison_rows_contains_file_chroma_pinecone(self):
        rows = build_backend_comparison_rows()

        self.assertEqual([row["backend"] for row in rows], ["file", "chroma", "pinecone"])

    def test_build_backend_comparison_rows_exposes_quantitative_metrics(self):
        rows = build_backend_comparison_rows()
        file_row = rows[0]
        chroma_row = rows[1]
        pinecone_row = rows[2]

        self.assertEqual(file_row["metadata_filter_operator_count"], 0)
        self.assertGreaterEqual(chroma_row["metadata_filter_operator_count"], 5)
        self.assertGreaterEqual(pinecone_row["metadata_filter_operator_count"], 5)
        self.assertGreater(file_row["manual_update_steps"], chroma_row["manual_update_steps"])
        self.assertEqual(chroma_row["persistent_restart_support"], 1)
        self.assertEqual(pinecone_row["persistent_restart_support"], 1)


if __name__ == "__main__":
    unittest.main()
