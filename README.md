# MoodTune

> ChromaDB와 Pinecone을 함께 사용해 감정 기반 음악 검색을 벡터 데이터베이스 시스템으로 확장한 개인 과제 프로젝트

## 1. 프로젝트 소개

MoodTune은 사용자의 자연어 질의를 OpenAI 임베딩으로 변환하고, 곡 설명과 메타데이터를 함께 저장한 벡터 데이터베이스에서 의미적으로 가까운 노래를 찾는 Streamlit 애플리케이션입니다.

이번 과제 버전에서는 기존의 `JSON/pickle + numpy` 기반 검색을 유지하면서, 같은 데이터셋을 ChromaDB와 Pinecone에도 저장해 다음 내용을 직접 비교할 수 있게 만들었습니다.

- 파일 기반 검색과 벡터 DB 검색의 차이
- ChromaDB의 자동 임베딩 방식과 Pinecone의 외부 벡터 주입 방식의 차이
- 하이브리드 검색과 메타데이터 필터링
- update, upsert, delete, idempotency
- 100 / 500 / 1000개 스케일 벤치마크

## 2. 선택한 벡터 DB와 선택 이유

### ChromaDB

- `PersistentClient`를 사용해 로컬 디스크에 영구 저장할 수 있습니다.
- 문서를 넣으면 컬렉션의 `embedding_function`이 자동으로 임베딩을 생성하는 흐름을 보여주기 좋습니다.
- 과제에서 요구한 `where` 기반 하이브리드 검색과 `update` / `upsert` 차이를 시연하기 좋습니다.

### Pinecone

- 서버리스 인덱스로 운영형 벡터 DB 흐름을 보여주기 좋습니다.
- 임베딩 벡터를 외부에서 생성한 뒤 `upsert` 하는 구조라서 ChromaDB와 접근 방식 비교가 분명합니다.
- namespace 기반 분리와 필터 기반 검색을 함께 보여줄 수 있습니다.

## 3. 사용 모델

- 임베딩 모델: `text-embedding-3-small`
- 생성 모델: `gpt-4.1-mini`

임베딩은 모두 `text-embedding-3-small`로 생성합니다.  
`gpt-4.1-mini`는 앱 내부 결과 요약과 Spotify 보조 기능에만 사용합니다.

## 4. 시스템 아키텍처

```text
data/songs.json
  -> vector_db/data.py
  -> vector_db/services.py
  -> init_db.py / vector_lab.py sync

OpenAI Embeddings API
  -> Chroma custom embedding_function
  -> Pinecone external vector upsert

ChromaDB PersistentClient
  -> cache/chromadb/
  -> collection: moodtune_base_all, moodtune_base_genre_*

Pinecone Serverless Index
  -> index: moodtune-song-vectors
  -> namespace: base_all, base_genre_*

Streamlit UI
  -> app.py
  -> Vector DB / DB Map / Benchmark

CLI / Demo
  -> init_db.py
  -> vector_lab.py
```

## 5. 데이터셋과 메타데이터 스키마

### 데이터셋

- 파일: `data/songs.json`
- 데이터 수: `1080`곡
- 도메인: 감정 기반 음악 추천
- 출처: 곡 제목, 아티스트, 장르, 감성 요약 가사를 직접 정리한 수작업 데이터셋

### 원본 필드

- `id`
- `title`
- `artist`
- `genre`
- `lyrics`
- `mood_tags`
- `youtube_music_url`

### 벡터 DB 저장 메타데이터

`vector_db/data.py`에서 아래 메타데이터를 함께 생성합니다.

- `song_id`
- `title`
- `artist`
- `genre`
- `genre_key`
- `primary_mood`
- `mood_tags`
- `lyrics_length`
- `title_length`
- `mood_count`
- `tenant`
- `dataset_name`
- `variant_index`
- `source_song_id`
- `has_multiple_moods`

과제의 최소 조건인 메타데이터 2개를 넘어서, 하이브리드 검색과 다중 컬렉션 또는 namespace 분리를 위해 여러 필드를 함께 저장합니다.

## 6. 핵심 요구사항 충족 방식

### A. 벡터 DB 구축

- ChromaDB: `vector_db/store.py`의 `ChromaSongStore`가 `chromadb.PersistentClient`를 사용합니다.
- Pinecone: `vector_db/store.py`의 `PineconeSongStore.ensure_index()`가 서버리스 인덱스를 생성합니다.
- 차원 지정:
  - `text-embedding-3-small` -> `1536`
  - `text-embedding-3-large` -> `3072`
- 현재 데이터셋은 `1080`곡으로 200개 이상 조건을 충족합니다.

### B. CRUD

- Create: `sync_backend()`와 `init_db.py`
- Read:
  - ID 조회: `get_backend_record()`
  - 의미 검색: `query_backend()`
- Update / Upsert:
  - Chroma: `run_chroma_update_vs_upsert_demo()`
    - `update_record()`는 기존 ID만 수정하고 새 ID를 만들지 않습니다.
    - `sync_records(..., skip_existing=False)`는 upsert 경로로 동작해 누락 ID를 삽입할 수 있습니다.
  - Pinecone: `run_pinecone_idempotency_demo()`
    - 같은 upsert를 반복 실행해도 namespace count와 ID 목록이 유지되는지 확인합니다.
- Delete:
  - ID 삭제: `delete_backend_records(..., record_id=...)`
  - 조건 삭제: `delete_backend_records(..., genre=..., primary_mood=...)`

### C. 하이브리드 검색

`vector_db/workflows.py`에 3개 시나리오를 구현했습니다.

- `focus_indie_semantic_filter`
- `rain_or_ballad_filter`
- `travel_genre_match`

사용 연산자:

- `$eq`
- `$ne`
- `$gte`
- `$and`
- `$or`

동일한 자연어 질의에 대해 순수 시맨틱 검색과 하이브리드 검색 결과를 나란히 비교하는 데모는 `Vector DB` 페이지와 `vector_lab.py hybrid` 명령으로 확인할 수 있습니다.

### D. 파일 기반 vs 벡터 DB 비교

`vector_db/services.py`의 `run_benchmark()`와 `vector_db/comparison.py`에서 다음을 비교합니다.

- 검색 속도
- 데이터 규모 변화에 따른 성능
- 메타데이터 필터링 가능 여부
- 데이터 추가 및 수정 복잡도
- 영속성 처리 방식

## 7. 확장 기능 구현

이번 과제의 확장 기능 7가지를 모두 구현했습니다.

1. ChromaDB와 Pinecone 동시 구현 및 비교
2. OpenAI 기반 Chroma custom embedding function
3. 다중 컬렉션과 namespace 분리
4. upsert 멱등성 자동 시연
5. 벡터 DB 기반 t-SNE / UMAP 시각화
6. Streamlit UI 구현
7. 100 / 500 / 1000 벤치마크와 대규모 데이터 확장

## 8. 프로젝트 구조

```text
MoodTune/
├── app.py
├── init_db.py
├── vector_lab.py
├── vector_db/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── comparison.py
│   ├── data.py
│   ├── filters.py
│   ├── lab.py
│   ├── services.py
│   ├── store.py
│   ├── ui.py
│   ├── visualizer.py
│   └── workflows.py
├── data/
│   └── songs.json
├── cache/
│   ├── chromadb/
│   ├── embeddings.pkl
│   ├── enriched_embeddings.pkl
│   └── mood_labels.pkl
├── tests/
├── docs/
├── .env.example
├── .gitignore
└── requirements.txt
```

## 9. 실행 방법

### 1. 가상환경과 의존성 설치

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 환경변수 설정

```powershell
Copy-Item .env.example .env
```

`.env`에 아래 값을 입력합니다.

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_SUMMARY_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=moodtune-song-vectors
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

### 3. 파일 기반 임베딩 캐시 생성

```powershell
python embed_songs.py
```

### 4. 벡터 DB 초기화

기본 초기화:

```powershell
python init_db.py --backend both
```

이미 저장된 ID는 기본적으로 재임베딩하지 않습니다.  
강제로 다시 upsert 하려면 아래처럼 실행합니다.

```powershell
python init_db.py --backend both --force-upsert
```

### 5. 앱 실행

```powershell
streamlit run app.py
```

OpenAI 연결이 불안정한 경우 검색 화면은 종료되지 않고 keyword fallback 결과로 이어집니다.

## 10. CLI 사용 예시

### 동기화

```powershell
python vector_lab.py sync --backend chroma
python vector_lab.py sync --backend pinecone
python vector_lab.py sync --backend both
```

### CRUD

```powershell
python vector_lab.py get --backend chroma --id song_001
python vector_lab.py query --backend pinecone --query "비 오는 밤 감성 노래" --genre 발라드
python vector_lab.py delete --backend chroma --id demo_missing_record
python vector_lab.py delete --backend pinecone --genre 인디 --primary-mood 집중
```

### Update / Upsert / Idempotency

```powershell
python vector_lab.py chroma-demo
python vector_lab.py pinecone-demo
```

### Hybrid Search

```powershell
python vector_lab.py hybrid --backend chroma --top-k 3
python vector_lab.py hybrid --backend pinecone --top-k 3
```

### Visualization

```powershell
python vector_lab.py map --backend chroma --method tsne --color-by genre
python vector_lab.py map --backend pinecone --method umap --color-by primary_mood
```

### Benchmark

```powershell
python vector_lab.py benchmark --query "비 오는 날 카페에서 듣는 노래" --runs 3
python vector_lab.py compare-matrix
```

## 11. Streamlit UI

앱에서 아래 과제용 화면을 사용할 수 있습니다.

- `Vector DB`
  - 동기화
  - ID 조회
  - 의미 검색
  - 삭제
  - 순수 시맨틱 검색 vs 하이브리드 검색 비교
  - Chroma update vs upsert 데모
  - Pinecone idempotent upsert 데모
- `DB Map`
  - 벡터 DB에서 임베딩을 읽어와 t-SNE / UMAP 시각화
- `Benchmark`
  - 파일 기반, ChromaDB, Pinecone의 100 / 500 / 1000 스케일 성능 비교

결과 카드에는 메타데이터와 `score`, `distance`가 함께 표시됩니다.

## 12. 영속성, 비용 관리, 중복 처리

- ChromaDB 영속 저장 위치: `cache/chromadb/`
- Pinecone 저장 위치: 서버리스 인덱스 `moodtune-song-vectors`
- `init_db.py`와 `sync_backend()`는 기본적으로 이미 저장된 ID를 재임베딩하지 않습니다.
- 멱등성 시연이 필요한 경우에만 `--force-upsert` 또는 데모 명령을 사용합니다.

`.env`, `cache/`, `cache/chromadb/`, `*.pkl`은 `.gitignore`에 등록되어 있습니다.

## 13. 검증 방법

```powershell
python -m unittest discover -s tests -q
python vector_lab.py compare-matrix
```

현재 테스트 스위트는 `70개`이며, 벡터 DB 모듈과 Streamlit 페이지까지 포함해 검증합니다.

## 14. 참고 자료

- Chroma Documentation
- Pinecone Documentation
- OpenAI Embeddings Documentation
- ANN Wikipedia
- Idempotency Wikipedia

## 15. AI 도구 사용 고지

구현 구조 정리, 테스트 보강, README 정리 과정에서 AI 도구를 보조적으로 활용했습니다.  
최종 제출 전에는 코드 실행과 테스트를 통해 실제 동작을 직접 검증했습니다.
