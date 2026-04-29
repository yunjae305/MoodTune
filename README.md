# MoodTune

> OpenAI 임베딩을 이용해 감정 표현을 의미 공간으로 바꾸고, 그와 가장 가까운 노래를 찾는 시맨틱 음악 검색 프로젝트

## 1. 프로젝트 소개

MoodTune은 사용자의 자연어 감정 표현을 입력받아, 가사와 메타데이터를 임베딩한 뒤 의미적으로 가장 가까운 곡을 추천하는 Streamlit 기반 AI 애플리케이션입니다.  
단순 키워드 일치가 아니라 벡터 공간에서의 유사도를 이용해 검색하며, 같은 원리를 제로샷 무드 분류에도 활용합니다.

이 프로젝트에서 OpenAI 모델은 두 가지 역할로 분리되어 있습니다.

- 임베딩 생성: `text-embedding-3-small`
- 생성 기능: `gpt-4.1-mini`

기본 검색 모드는 로컬 임베딩 기반 시맨틱 검색입니다. `Spotify 추천` 토글을 켜면 임베딩 검색과 무드 분류를 비활성화하고, Spotify API 기반 추천만 사용합니다.

## 2. 과제 요구사항 충족 여부

### 핵심 요구사항

| 항목 | 구현 내용 | 관련 파일 |
| --- | --- | --- |
| 임베딩 생성 및 활용 | 100곡 데이터셋을 `text-embedding-3-small`로 임베딩 | `embed_songs.py`, `data/songs.json` |
| 임베딩 캐싱 | `embeddings.pkl`, `enriched_embeddings.pkl`, `mood_labels.pkl`로 저장 | `embed_songs.py`, `classify.py`, `cache/` |
| 핵심 기능 1개 이상 | 시맨틱 검색 + 제로샷 무드 분류 구현 | `search.py`, `classify.py`, `app.py` |
| 코사인 유사도 직접 구현 | 수식 기반 직접 구현 + 배치 계산 | `cosine.py` |
| 라이브러리 결과 검증 | `scipy` 결과와 오차 검증 함수 포함 | `cosine.py` |

### 확장 기능

| 항목 | 구현 내용 | 관련 파일 |
| --- | --- | --- |
| t-SNE 시각화 | 임베딩을 2차원으로 축소해 무드 군집 시각화 | `tsne_visualizer.py`, `app.py` |
| 풍부한 임베딩 | 제목, 아티스트, 무드 태그, 장르를 가사와 결합한 enriched embedding 제공 | `embed_songs.py`, `app.py` |
| 복수 기능 구현 | 시맨틱 검색 + 제로샷 분류 동시 구현 | `search.py`, `classify.py` |
| 키워드 검색 비교 | TF-IDF 기반 키워드 검색과 의미 검색 결과 비교 | `keyword_search.py`, `app.py` |
| UI 구현 | Streamlit 웹 인터페이스 제공 | `app.py` |

## 3. 시스템 아키텍처

### 3단계 프로세스

1. 데이터셋 임베딩
   `data/songs.json`의 각 곡을 임베딩해 `cache/*.pkl`에 저장합니다.
2. 사용자 질의 임베딩
   사용자의 감정 표현을 같은 임베딩 공간의 벡터로 변환합니다.
3. 유사도 계산 및 결과 반환
   코사인 유사도로 가장 가까운 곡과 무드 레이블을 찾고, UI에 결과를 표시합니다.

### 데이터 흐름

```text
data/songs.json
  -> embed_songs.py
  -> cache/embeddings.pkl, cache/enriched_embeddings.pkl

사용자 질의
  -> app.py
  -> OpenAI embeddings.create()
  -> cosine.py
  -> search.py / classify.py
  -> Streamlit UI

검색 결과
  -> ai_summary.py
  -> gpt-4.1-mini
  -> 자연어 결과 요약
```

## 4. 프로젝트 구조

```text
MoodTune/
├── app.py
├── ai_summary.py
├── classify.py
├── cosine.py
├── embed_songs.py
├── keyword_search.py
├── search.py
├── spotify_api.py
├── spotify_mapper.py
├── tsne_visualizer.py
├── ui_reference.py
├── data/
│   └── songs.json
├── cache/
│   ├── embeddings.pkl
│   ├── enriched_embeddings.pkl
│   └── mood_labels.pkl
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## 5. 데이터셋 설명

- 데이터 수: 100곡
- 구성: 한국어 중심 데이터셋 + 일부 영어 곡
- 필드: `id`, `title`, `artist`, `lyrics`, `mood_tags`, `genre`, `youtube_music_url`
- 목적: 감정 기반 질의에 대해 의미 유사도 검색이 가능하도록 구성

### 데이터 출처 및 가공 방식

- 곡 제목, 아티스트, 장르 정보는 공개적으로 알려진 음악 메타데이터를 참고해 직접 정리했습니다.
- `lyrics` 필드는 원문 가사를 대량 수집한 데이터셋이 아니라, 검색 실험과 임베딩 비교를 위해 직접 구성한 감성 요약 텍스트입니다.
- `youtube_music_url`은 결과 확인용 링크입니다.

## 6. 사용 모델과 기능 분리

### 임베딩 모델

- 모델명: `text-embedding-3-small`
- 사용 위치:
  - 곡 데이터셋 임베딩 생성
  - 사용자 질의 임베딩
  - 제로샷 무드 레이블 임베딩

### 생성 모델

- 모델명: `gpt-4.1-mini`
- 사용 위치:
  - 검색 결과를 자연어로 짧게 설명하는 요약 카드
  - Spotify 검색용 장르/영문 키워드 생성

즉, 이 프로젝트에서 임베딩은 GPT 모델로 만드는 것이 아니라 OpenAI 임베딩 전용 모델로 생성합니다. `gpt-4.1-mini`는 결과 설명과 Spotify 검색 키워드 생성에 사용됩니다.

## 7. 캐싱 전략

과제 요구사항에 맞춰 매번 같은 데이터셋을 다시 임베딩하지 않도록 파일 캐싱을 구현했습니다.

- 곡 기본 임베딩 캐시: `cache/embeddings.pkl`
- 풍부한 임베딩 캐시: `cache/enriched_embeddings.pkl`
- 무드 레이블 임베딩 캐시: `cache/mood_labels.pkl`

앱 실행 중 반복 질의에 대해서는 Streamlit 캐시도 함께 사용합니다.

- `@st.cache_data`: 질의 임베딩, 결과 요약, 캐시 파일 로딩
- `@st.cache_resource`: OpenAI 클라이언트

## 8. 실행 방법

### 1. 가상환경 생성 및 패키지 설치

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 환경변수 설정

```powershell
Copy-Item .env.example .env
```

`.env`에 아래 값을 채워 넣습니다.

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_SUMMARY_MODEL=gpt-4.1-mini
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

- `OPENAI_API_KEY`: 필수
- `OPENAI_SUMMARY_MODEL`: 결과 요약 모델 지정용, 기본값은 코드에도 `gpt-4.1-mini`로 설정
- Spotify 관련 값: Spotify 기능을 쓸 때만 필요

### 3. 임베딩 캐시 생성

```powershell
python embed_songs.py
```

처음 한 번만 실행하면 되고, 이후에는 `cache/`의 `.pkl` 파일을 재사용합니다.

### 4. 앱 실행

```powershell
streamlit run app.py
```

## 9. 주요 기능

### 시맨틱 검색

- 사용자의 감정 표현을 임베딩한 뒤, 데이터셋에서 코사인 유사도가 높은 곡을 Top-K로 반환합니다.

### 제로샷 무드 분류

- 8개의 풍부한 무드 라벨을 임베딩해 두고, 질의를 가장 가까운 라벨로 자동 분류합니다.

### 키워드 검색 비교

- TF-IDF 기반 키워드 검색 결과를 함께 보여 주어 시맨틱 검색과 비교할 수 있습니다.

### 풍부한 임베딩 비교

- 단순 가사 임베딩과 enriched embedding 결과를 나란히 비교할 수 있습니다.

### t-SNE 시각화

- 임베딩 공간을 2차원으로 축소해 무드 분포와 질의 위치를 시각화합니다.

### Spotify 추천 모드

- Spotify 토글을 켜면 로컬 임베딩 검색과 무드 분류를 끄고 Spotify 추천 결과만 표시합니다.
- Spotify 추천은 deprecated 된 recommendations 엔드포인트 대신, `gpt-4.1-mini`가 만든 장르/검색 키워드와 질의 기반 검색 조합으로 동작합니다.
- 이 모드에서는 비교 분석과 무드 맵 같은 임베딩 전용 화면도 비활성화됩니다.

## 10. 코사인 유사도 구현

`cosine.py`에는 다음 내용이 포함되어 있습니다.

- 코사인 유사도 직접 구현 `cosine_similarity`
- 검색 최적화를 위한 배치 계산 `cosine_similarity_batch`
- `scipy` 결과와의 일치 여부 검증 `verify_against_scipy`

수식은 아래와 같습니다.

```text
cos(theta) = (A · B) / (||A|| ||B||)
```

## 11. 검증 방법

### 기본 검증

```powershell
python -m py_compile app.py ai_summary.py classify.py cosine.py embed_songs.py keyword_search.py search.py tsne_visualizer.py
python -m unittest discover -s tests -q
```

### 주요 테스트 범위

- `.env`에서 API 키와 모델 설정 로딩
- Spotify 모드에서 임베딩 경로 비활성화
- UI 상태 전환
- 검색 결과 비교 로직
- 코사인 유사도 직접 구현 검증

## 12. 보고서와 데모 영상에 바로 쓸 수 있는 포인트

### 데모에서 강조할 점

- 키워드는 겹치지 않아도 의미적으로 비슷한 질의가 검색되는 사례
- 키워드 검색과 시맨틱 검색 결과 차이
- 제로샷 무드 분류 결과와 유사도 점수
- t-SNE에서 무드 군집이 어떻게 보이는지
- Spotify 모드와 로컬 임베딩 모드의 차이

### 보고서에서 설명하면 좋은 설계 선택

- 왜 `text-embedding-3-small`을 선택했는지
- 왜 enriched embedding을 따로 비교했는지
- 왜 코사인 유사도를 직접 구현했는지
- 어떤 질의에서 성능이 좋고, 어떤 질의에서 한계가 있는지

## 13. 보안 및 제출 주의사항

- API 키는 코드에 하드코딩하지 않았습니다.
- 실제 런타임 값은 루트 `.env`에서 읽고, `.env.example`은 템플릿입니다.
- `.gitignore`에 `.env`, `cache/`, `*.pkl`이 포함되어 있어 비밀값과 캐시 파일이 기본적으로 제외됩니다.

## 14. AI 도구 사용 내역

- 구현 구조 정리, 테스트 보강, README 정리 과정에서 AI 도구를 보조적으로 활용했습니다.
- 최종 제출 전 코드 구조와 동작, 테스트 결과를 직접 확인했습니다.
