import unittest

from vector_db.store import (
    ChromaOpenAIEmbeddingFunction,
    embedding_dimension_for_model,
    filter_missing_records,
    group_records_by_scope,
    prepare_pinecone_vectors,
)


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[float(len(text))] for text in texts]


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "id": "song_001",
                "document": "focus song",
                "metadata": {
                    "genre_key": "genre_a",
                    "genre": "인디",
                    "song_id": "song_001",
                },
            },
            {
                "id": "song_002",
                "document": "rain song",
                "metadata": {
                    "genre_key": "genre_b",
                    "genre": "발라드",
                    "song_id": "song_002",
                },
            },
        ]

    def test_embedding_dimension_for_model_matches_openai_dimensions(self):
        self.assertEqual(embedding_dimension_for_model("text-embedding-3-small"), 1536)
        self.assertEqual(embedding_dimension_for_model("text-embedding-3-large"), 3072)

    def test_chroma_openai_embedding_function_uses_supplied_embedder(self):
        embedding_function = ChromaOpenAIEmbeddingFunction(FakeEmbedder())

        vectors = embedding_function(["alpha", "beta"])

        self.assertEqual(vectors, [[5.0], [4.0]])

    def test_group_records_by_scope_creates_all_and_genre_partitions(self):
        grouped = group_records_by_scope(self.records)

        self.assertEqual(set(grouped.keys()), {"all", "genre_a", "genre_b"})
        self.assertEqual(len(grouped["all"]), 2)
        self.assertEqual(grouped["genre_a"][0]["id"], "song_001")

    def test_prepare_pinecone_vectors_preserves_id_values_and_metadata(self):
        vectors = prepare_pinecone_vectors(self.records, [[0.1, 0.2], [0.3, 0.4]])

        self.assertEqual(vectors[0]["id"], "song_001")
        self.assertEqual(vectors[0]["values"], [0.1, 0.2])
        self.assertEqual(vectors[1]["metadata"]["genre"], "발라드")

    def test_filter_missing_records_skips_existing_ids(self):
        filtered = filter_missing_records(self.records, {"song_002"})

        self.assertEqual([record["id"] for record in filtered], ["song_001"])


if __name__ == "__main__":
    unittest.main()
