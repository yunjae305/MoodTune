# MoodTune Streamlit Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ZIP의 구조와 디자인을 현재 Streamlit 앱에 맞게 재구성해 실행 가능한 MoodTune UI를 만든다.

**Architecture:** 기존 검색 로직은 유지하고 UI 레이어만 다시 설계한다. `ui_reference.py`에 ZIP 기준 토큰과 프레젠테이션 헬퍼를 두고, `app.py`는 상태 전환과 렌더링에 집중한다.

**Tech Stack:** Python 3.11, Streamlit, OpenAI Embeddings, NumPy, matplotlib

---

### Task 1: ZIP UI 기준값 정리

**Files:**
- Create: `ui_reference.py`
- Create: `tests/test_ui_reference.py`

- [ ] ZIP의 무드 색상, 네비 항목, 칩 문구를 Python 상수로 정리한다.
- [ ] 결과 설명과 비교 행 생성 헬퍼를 만든다.
- [ ] `python -m unittest tests.test_ui_reference -v`로 테스트를 확인한다.

### Task 2: Streamlit 화면 상태 재구성

**Files:**
- Modify: `app.py`

- [ ] `home`, `matching`, `results`, `map`, `compare` 뷰 상태를 `st.session_state`에 추가한다.
- [ ] 검색 시작과 검색 완료 흐름을 분리해 매칭 화면을 거치도록 바꾼다.
- [ ] ZIP 기준 좌측 네비와 본문 캔버스 레이아웃을 추가한다.

### Task 3: ZIP 스타일과 결과 화면 이식

**Files:**
- Modify: `app.py`

- [ ] ZIP `styles.css` 기준 토큰과 타이포를 Streamlit CSS로 옮긴다.
- [ ] 검색 홈, 결과 헤더, 무드 pill, 결과 row, 요약 카드, 비교 섹션을 이식한다.
- [ ] 기존 기능인 키워드 비교와 단순/풍부 임베딩 비교를 새 구조에 맞게 재배치한다.

### Task 4: 검증

**Files:**
- Modify: `app.py`

- [ ] `python -m unittest discover -s tests -v`를 실행한다.
- [ ] `python -m py_compile app.py ui_reference.py search.py keyword_search.py classify.py embed_songs.py tsne_visualizer.py`를 실행한다.
- [ ] `streamlit run app.py --server.headless true`로 실행 여부를 확인한다.
