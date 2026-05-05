import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


CHROMA_DIR = Path("cache/chromadb")
DEFAULT_CHROMA_PREFIX = "moodtune"
DEFAULT_PINECONE_INDEX = "moodtune-song-vectors"
DEFAULT_PINECONE_CLOUD = "aws"
DEFAULT_PINECONE_REGION = "us-east-1"


def embedding_dimension_for_model(model: str) -> int:
    if model == "text-embedding-3-large":
        return 3072
    return 1536


def group_records_by_scope(records: list[dict]) -> dict[str, list[dict]]:
    grouped = {"all": list(records)}
    for record in records:
        scope = record["metadata"].get("genre_key", "genre_unknown")
        grouped.setdefault(scope, []).append(record)
    return grouped


def prepare_pinecone_vectors(records: list[dict], embeddings: list[list[float]]) -> list[dict]:
    if len(records) != len(embeddings):
        raise ValueError("records and embeddings must have the same length")
    vectors = []
    for record, embedding in zip(records, embeddings):
        vectors.append(
            {
                "id": record["id"],
                "values": embedding,
                "metadata": record["metadata"],
            }
        )
    return vectors


def filter_missing_records(records: list[dict], existing_ids: set[str]) -> list[dict]:
    return [record for record in records if record["id"] not in existing_ids]


def normalize_chroma_results(response: dict) -> list[dict]:
    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]
    rows = []
    for record_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        row = dict(metadata or {})
        row["id"] = record_id
        row["document"] = document
        row["distance"] = float(distance)
        row["score"] = 1.0 / (1.0 + float(distance))
        rows.append(row)
    return rows


def normalize_pinecone_matches(matches: list[dict]) -> list[dict]:
    rows = []
    for match in matches:
        row = dict(match.get("metadata") or {})
        row["id"] = match.get("id")
        row["score"] = float(match.get("score", 0.0))
        rows.append(row)
    return rows


class OpenAIEmbedder:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        load_dotenv()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.model = model or os.environ.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
        self.client = OpenAI(api_key=self.api_key)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        embeddings = []
        for index in range(0, len(texts), batch_size):
            batch = texts[index : index + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
            if index + batch_size < len(texts):
                time.sleep(0.2)
        return embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class ChromaOpenAIEmbeddingFunction:
    def __init__(self, embedder: OpenAIEmbedder):
        self.embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embedder.embed_texts(list(input))

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return "moodtune_openai"

    @staticmethod
    def build_from_config(config: dict) -> "ChromaOpenAIEmbeddingFunction":
        model = config.get("model")
        return ChromaOpenAIEmbeddingFunction(OpenAIEmbedder(model=model))

    def get_config(self) -> dict:
        return {"model": self.embedder.model}

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]


class ChromaSongStore:
    def __init__(
        self,
        persist_dir: Path | str = CHROMA_DIR,
        collection_prefix: str = DEFAULT_CHROMA_PREFIX,
        embedder: OpenAIEmbedder | None = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_prefix = collection_prefix
        self.embedder = embedder or OpenAIEmbedder()
        self._client = None

    def _load_module(self):
        import chromadb

        return chromadb

    def client(self):
        if self._client is None:
            chromadb = self._load_module()
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def collection_name(self, scope: str) -> str:
        return f"{self.collection_prefix}_{scope}"

    def collection(self, scope: str = "all"):
        return self.client().get_or_create_collection(
            name=self.collection_name(scope),
            embedding_function=ChromaOpenAIEmbeddingFunction(self.embedder),
        )

    def sync_records(self, records: list[dict], skip_existing: bool = True) -> dict[str, int]:
        """기존 ID는 건너뛰고 필요한 문서만 Chroma 컬렉션에 적재한다."""
        grouped = group_records_by_scope(records)
        counts = {}
        for scope, scope_records in grouped.items():
            collection = self.collection(scope)
            records_to_sync = scope_records
            if skip_existing:
                existing = collection.get(ids=[record["id"] for record in scope_records])
                existing_ids = set(existing.get("ids", []))
                records_to_sync = filter_missing_records(scope_records, existing_ids)
            if records_to_sync:
                collection.upsert(
                    ids=[record["id"] for record in records_to_sync],
                    documents=[record["document"] for record in records_to_sync],
                    metadatas=[record["metadata"] for record in records_to_sync],
                )
            counts[scope] = collection.count()
        return counts

    def reset_scopes(self, scopes: list[str]) -> None:
        client = self.client()
        for scope in scopes:
            try:
                client.delete_collection(name=self.collection_name(scope))
            except Exception:
                pass

    def get_by_id(self, record_id: str, scope: str = "all") -> dict:
        """ID 하나를 직접 조회해 문서, 메타데이터, 임베딩을 함께 확인한다."""
        return self.collection(scope).get(
            ids=[record_id],
            include=["documents", "metadatas", "embeddings"],
        )

    def query(self, query_text: str, top_k: int = 5, scope: str = "all", where: dict | None = None) -> list[dict]:
        """텍스트 질의와 where 필터를 함께 적용해 Chroma 하이브리드 검색을 수행한다."""
        response = self.collection(scope).query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return normalize_chroma_results(response)

    def update_record(self, record_id: str, scope: str = "all", document: str | None = None, metadata: dict | None = None) -> dict:
        """기존 ID에만 문서 또는 메타데이터를 반영하고, 없는 ID는 새로 만들지 않는다."""
        payload = {"ids": [record_id]}
        if document is not None:
            payload["documents"] = [document]
        if metadata is not None:
            payload["metadatas"] = [metadata]
        self.collection(scope).update(**payload)
        return self.get_by_id(record_id, scope=scope)

    def delete_records(self, scope: str = "all", ids: list[str] | None = None, where: dict | None = None) -> None:
        """ID 직접 삭제와 where 조건 삭제를 같은 인터페이스로 수행한다."""
        payload = {}
        if ids:
            payload["ids"] = ids
        if where:
            payload["where"] = where
        self.collection(scope).delete(**payload)

    def fetch_all(self, scope: str = "all") -> dict:
        return self.collection(scope).get(include=["documents", "metadatas", "embeddings"])


class PineconeSongStore:
    def __init__(
        self,
        index_name: str | None = None,
        cloud: str = DEFAULT_PINECONE_CLOUD,
        region: str = DEFAULT_PINECONE_REGION,
        embedder: OpenAIEmbedder | None = None,
        namespace_prefix: str = "",
    ):
        load_dotenv()
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME") or DEFAULT_PINECONE_INDEX
        self.cloud = cloud
        self.region = region
        self.embedder = embedder or OpenAIEmbedder()
        self.namespace_prefix = namespace_prefix.strip("_")
        self.api_key = os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required")
        self._client = None
        self._index = None

    def _load_module(self):
        from pinecone import Pinecone, ServerlessSpec

        return Pinecone, ServerlessSpec

    def client(self):
        if self._client is None:
            Pinecone, _ = self._load_module()
            self._client = Pinecone(api_key=self.api_key)
        return self._client

    def ensure_index(self):
        if self._index is not None:
            return self._index
        client = self.client()
        _, ServerlessSpec = self._load_module()
        existing_names = [item["name"] if isinstance(item, dict) else item.name for item in client.list_indexes()]
        if self.index_name not in existing_names:
            client.create_index(
                name=self.index_name,
                dimension=embedding_dimension_for_model(self.embedder.model),
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        self._index = client.Index(self.index_name)
        return self._index

    def namespace(self, scope: str) -> str:
        if not self.namespace_prefix:
            return scope
        return f"{self.namespace_prefix}_{scope}"

    def sync_records(self, records: list[dict], skip_existing: bool = True) -> dict[str, int]:
        """기존 ID는 재임베딩하지 않고 필요한 벡터만 Pinecone 네임스페이스에 upsert한다."""
        index = self.ensure_index()
        grouped = group_records_by_scope(records)
        ids_to_embed = set()
        records_to_sync_by_scope = {}
        for scope, scope_records in grouped.items():
            records_to_sync = scope_records
            if skip_existing:
                existing_ids = set(self.list_ids(scope=scope))
                records_to_sync = filter_missing_records(scope_records, existing_ids)
            records_to_sync_by_scope[scope] = records_to_sync
            ids_to_embed.update(record["id"] for record in records_to_sync)
        unique_records = {record["id"]: record for record in records}
        embeddings_by_id = {}
        if ids_to_embed:
            records_to_embed = [unique_records[record_id] for record_id in ids_to_embed]
            embeddings = self.embedder.embed_texts([record["document"] for record in records_to_embed])
            embeddings_by_id = {
                record["id"]: embedding
                for record, embedding in zip(records_to_embed, embeddings)
            }
        counts = {}
        for scope, scope_records in grouped.items():
            namespace = self.namespace(scope)
            records_to_sync = records_to_sync_by_scope[scope]
            if records_to_sync:
                vectors = prepare_pinecone_vectors(
                    records_to_sync,
                    [embeddings_by_id[record["id"]] for record in records_to_sync],
                )
                for start in range(0, len(vectors), 100):
                    batch = vectors[start : start + 100]
                    index.upsert(vectors=batch, namespace=namespace)
            counts[scope] = len(list(self.list_ids(scope=scope)))
        return counts

    def reset_scopes(self, scopes: list[str]) -> None:
        index = self.ensure_index()
        for scope in scopes:
            ids = self.list_ids(scope=scope)
            if not ids:
                continue
            namespace = self.namespace(scope)
            for start in range(0, len(ids), 100):
                index.delete(ids=ids[start : start + 100], namespace=namespace)

    def list_ids(self, scope: str = "all") -> list[str]:
        ids = []
        for page in self.ensure_index().list(namespace=self.namespace(scope)):
            ids.extend(page)
        return ids

    def get_by_id(self, record_id: str, scope: str = "all") -> dict:
        """namespace 안의 벡터를 ID로 직접 조회한다."""
        return self.ensure_index().fetch(ids=[record_id], namespace=self.namespace(scope))

    def query(self, query_text: str, top_k: int = 5, scope: str = "all", filter: dict | None = None) -> list[dict]:
        """외부 임베딩 벡터와 metadata filter를 함께 써서 Pinecone 검색을 수행한다."""
        vector = self.embedder.embed_text(query_text)
        response = self.ensure_index().query(
            namespace=self.namespace(scope),
            vector=vector,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
            include_values=False,
        )
        if hasattr(response, "matches"):
            matches = response.matches
        else:
            matches = response.get("matches", [])
        return normalize_pinecone_matches(matches)

    def update_record(self, record_id: str, scope: str = "all", metadata: dict | None = None, values: list[float] | None = None) -> dict:
        """기존 벡터 ID의 metadata 또는 values를 갱신한다."""
        payload = {"namespace": self.namespace(scope), "id": record_id}
        if metadata is not None:
            payload["set_metadata"] = metadata
        if values is not None:
            payload["values"] = values
        self.ensure_index().update(**payload)
        return self.get_by_id(record_id, scope=scope)

    def delete_records(self, scope: str = "all", ids: list[str] | None = None, filter: dict | None = None) -> None:
        """ID 삭제와 metadata filter 삭제를 Pinecone namespace 단위로 수행한다."""
        payload = {"namespace": self.namespace(scope)}
        if ids:
            payload["ids"] = ids
        if filter:
            payload["filter"] = filter
        self.ensure_index().delete(**payload)

    def fetch_all(self, scope: str = "all") -> list[dict]:
        ids = self.list_ids(scope=scope)
        if not ids:
            return []
        rows = []
        for start in range(0, len(ids), 1000):
            batch_ids = ids[start : start + 1000]
            response = self.ensure_index().fetch(ids=batch_ids, namespace=self.namespace(scope))
            if hasattr(response, "vectors"):
                vectors = response.vectors
            else:
                vectors = response.get("vectors", {})
            rows.extend(vectors.values())
        return rows
