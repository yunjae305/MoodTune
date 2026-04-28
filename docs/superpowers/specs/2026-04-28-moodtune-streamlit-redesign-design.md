# MoodTune Streamlit Redesign

**Goal:** ZIP에 있는 MoodTune의 구조와 시각 톤을 현재 Streamlit 앱 틀 위에 재구성한다.

**Architecture:** 검색 엔진과 임베딩 로직은 기존 Python 모듈을 유지한다. 화면은 `home`, `matching`, `results`, `map`, `compare` 다섯 뷰로 나누고 `st.session_state`로 전환한다. ZIP의 `styles.css`, `common.jsx`, `search.jsx`, `results.jsx`를 기준으로 색상 토큰, 좌측 네비, 검색 히어로, 결과 리스트, 비교 화면을 Streamlit에서 재현한다.

**Scope**
- 좌측 고정 네비와 메인 캔버스 구조 적용
- 검색 홈을 ZIP의 히어로 스타일로 재구성
- 매칭 중간 화면 추가
- 결과 헤더, 무드 pill, 결과 row, 요약 카드 추가
- 무드 맵과 비교 분석 뷰 분리
- 기존 시맨틱 검색, 키워드 비교, t-SNE, 무드 분류 로직 재사용

**Constraints**
- 실행 진입점은 계속 `app.py`
- 사용자 제공 ZIP의 구조와 디자인을 우선 참조
- 새 코드에는 주석을 추가하지 않음
- 실제 데이터셋은 현재 `data/songs.json`을 사용
