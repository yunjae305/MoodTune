# MoodTune 🎵

> "비 오는 날 혼자 듣기 좋은 노래" — 자연어 감성 표현으로 가사 임베딩 기반 시맨틱 검색

## 프로젝트 개요

기존 음악 플랫폼의 키워드 매칭 검색과 달리, MoodTune은 가사 전문을 `text-embedding-3-small`로 임베딩하여 **의미론적 유사도**로 곡을 검색합니다.  
키워드가 단 하나도 겹치지 않아도 감성이 맞는 곡을 반환하는 것이 핵심 목표입니다.

## 파일 구조

```
MoodTune/
├── app.py                    # Streamlit 메인 앱
├── cosine.py                 # 코사인 유사도 직접 구현 + scipy 검증
├── search.py                 # 시맨틱 검색 로직
├── classify.py               # 제로샷 무드 분류
├── keyword_search.py         # TF-IDF 키워드 검색 (비교용)
├── tsne_visualizer.py        # t-SNE 2D 시각화
├── embed_songs.py            # 임베딩 생성 + pickle 캐싱
├── data/
│   └── songs.json            # 100곡 메타데이터 + YouTube Music URL
├── cache/                    # 임베딩 캐시 (자동 생성)
│   ├── embeddings.pkl
│   ├── enriched_embeddings.pkl
│   └── mood_labels.pkl
├── .env                      # API 키 (gitignore)
├── .env.example
├── requirements.txt
└── README.md
```

## 빠른 시작

### 1. 가상환경 및 패키지 설치

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력
```

### 3. 임베딩 생성 (최초 1회)

```bash
python embed_songs.py
# 예상 비용: $0.002 미만 / 예상 시간: 30초~1분
```

### 4. 앱 실행

```bash
streamlit run app.py
```

## 기술 스택

| 역할        | 기술                              |
| ----------- | --------------------------------- |
| 임베딩 모델 | `text-embedding-3-small` (OpenAI) |
| 캐싱        | pickle (.pkl)                     |
| 유사도 계산 | numpy 직접 구현 + scipy 검증      |
| 시각화      | scikit-learn (t-SNE) + matplotlib |
| 키워드 검색 | scikit-learn TF-IDF               |
| UI          | Streamlit                         |
| 데이터      | JSON (100곡, 한국어 80 + 영어 20) |

## 주요 기능

- **시맨틱 검색**: 자연어 감성 → 코사인 유사도 → Top-K 반환
- **제로샷 무드 분류**: 8개 카테고리 풍부한 레이블 임베딩으로 분류
- **코사인 유사도 직접 구현**: 수식 단계별 구현 + scipy 오차 < 1e-6 검증
- **t-SNE 시각화**: 임베딩 2D 축소 + 무드 클러스터 + 쿼리 위치 표시
- **키워드 vs 시맨틱 비교**: TF-IDF와 나란히 비교, 키워드 0개 겹침 사례 분석
- **풍부한 임베딩**: 제목+아티스트+무드태그+가사 결합 임베딩 지원

## 데이터셋

- 총 100곡: 한국어 80곡 + 영어 20곡
- 장르: 발라드, 인디, 팝, R&B, 힙합
- 무드 카테고리: 새벽 감성 / 비 오는 날 / 그리움·이별 / 설렘·사랑 / 위로·힐링 / 신남·파티 / 집중·공부 / 여행·자유
- 각 곡 항목: id, title, artist, lyrics, mood_tags, genre, youtube_music_url

## AI 도구 활용 내역

본 프로젝트 구조 설계 및 코드 구현에 Claude (Anthropic)를 활용하였습니다.
