# MoodTune Vector DB Extension

## Added Modules

- `vector_db/data.py`
- `vector_db/store.py`
- `vector_db/services.py`
- `vector_db/workflows.py`
- `vector_db/visualizer.py`
- `vector_db/benchmark.py`
- `vector_db/ui.py`
- `vector_db/lab.py`
- `vector_lab.py`

## Covered Requirements

- ChromaDB `PersistentClient`
- Pinecone serverless index
- OpenAI custom embedding function for ChromaDB
- Multiple Chroma collections and Pinecone namespaces
- Batch create, get, query, update, upsert, delete
- Chroma update vs upsert demo
- Pinecone idempotent upsert demo
- Hybrid search scenarios with metadata filters
- File cache vs vector DB comparison
- Vector DB backed t-SNE and UMAP visualization
- Streamlit UI pages for Vector DB search, map, and benchmark
- 100 / 500 / 1000 dataset benchmark
- `data/songs_expanded_1000.json` generated on demand and ignored from git

## Required Environment Variables

```env
OPENAI_API_KEY=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=moodtune-song-vectors
```

## CLI Examples

```powershell
python vector_lab.py sync --backend chroma
python vector_lab.py sync --backend pinecone
python vector_lab.py hybrid --backend chroma --top-k 3
python vector_lab.py hybrid --backend pinecone --top-k 3
python vector_lab.py chroma-demo
python vector_lab.py pinecone-demo
python vector_lab.py compare-matrix
python vector_lab.py map --backend chroma --method tsne --color-by genre
python vector_lab.py map --backend pinecone --method umap --color-by primary_mood
python vector_lab.py benchmark --query "비 오는 날 카페에서 듣는 노래" --runs 1
```

## Streamlit Pages

Run:

```powershell
streamlit run app.py
```

Then open:

- `Vector DB`
- `DB Map`
- `Benchmark`
