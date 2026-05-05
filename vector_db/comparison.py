def build_backend_comparison_rows() -> list[dict]:
    return [
        {
            "backend": "file",
            "storage_type": "json_pickle_numpy",
            "metadata_filter_operator_count": 0,
            "manual_update_steps": 4,
            "persistent_restart_support": 1,
            "ann_managed_by_backend": 0,
            "notes": "manual load, edit, re-embed, rewrite",
        },
        {
            "backend": "chroma",
            "storage_type": "persistent_vector_db",
            "metadata_filter_operator_count": 5,
            "manual_update_steps": 1,
            "persistent_restart_support": 1,
            "ann_managed_by_backend": 1,
            "notes": "PersistentClient with where filters and managed index",
        },
        {
            "backend": "pinecone",
            "storage_type": "serverless_vector_db",
            "metadata_filter_operator_count": 5,
            "manual_update_steps": 1,
            "persistent_restart_support": 1,
            "ann_managed_by_backend": 1,
            "notes": "serverless index with external vectors and filter query",
        },
    ]
