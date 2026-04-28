"""
MoodTune
"""

import os
import pickle
import time
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from classify import MOOD_CATEGORIES, load_or_create_mood_embeddings
from keyword_search import compare_search_results, keyword_search
from search import search_from_query_vector
from tsne_visualizer import get_tsne_coords_for_query
from image_fetcher import get_album_art_url
from spotify_api import SpotifyClient, get_spotify_auth_url, exchange_code_for_token
from spotify_mapper import map_mood_to_spotify_features
from ui_reference import (
    NAV_ITEMS,
    QUERY_CHIPS,
    build_compare_rows,
    build_result_reason,
    build_result_summary,
    build_search_state_update,
    consume_query_prefill,
    get_mood_theme,
    get_sidebar_moods,
    truncate_text,
)


load_dotenv(override=True)

CACHE_DIR = Path("cache")
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MOOD = "새벽 감성"
FIXED_TOP_K = 8

st.set_page_config(
    page_title="MoodTune",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="auto",
)


def with_alpha(color: str, alpha: str) -> str:
    if color.startswith("#"):
        return color # Simplification for hex colors
    return color.replace(")", f" / {alpha})")


def init_state() -> None:
    defaults = {
        "view": "home",
        "query_input": "",
        "query_prefill": "",
        "pending_query": "",
        "last_query": "",
        "top_k": FIXED_TOP_K,
        "use_enriched": True,
        "use_spotify": False,  # 기본값: 로컬 임베딩 검색 (Spotify Recommendations API deprecated)
        "music_region": "국내",  # 기본 지역 설정 (국내/국외)
        "enriched_results": [],
        "simple_results": [],
        "kw_results": [],
        "mood_ranking": [],
        "top_mood": DEFAULT_MOOD,
        "top_mood_score": 0.0,
        "query_vec": None,
        "result_summary": "",
        "comparison": None,
        "map_requested": False,
        "last_search_settings": {"top_k": FIXED_TOP_K},
        "spotify_access_token": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state["top_k"] = FIXED_TOP_K
    st.session_state["last_search_settings"] = {"top_k": FIXED_TOP_K}


@st.cache_resource
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
        st.stop()
    return OpenAI(api_key=api_key)


@st.cache_data
def load_embeddings_cache(enriched: bool = True) -> dict | None:
    path = CACHE_DIR / ("enriched_embeddings.pkl" if enriched else "embeddings.pkl")
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_mood_embeddings_cache() -> dict | None:
    path = CACHE_DIR / "mood_labels.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def embed_text(text: str) -> np.ndarray:
    client = get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return np.array(response.data[0].embedding)


def current_theme() -> dict:
    return get_mood_theme(st.session_state.get("top_mood", DEFAULT_MOOD))


def active_results() -> list[dict]:
    if st.session_state["use_enriched"]:
        return st.session_state.get("enriched_results", [])
    return st.session_state.get("simple_results", [])


def apply_theme(theme: dict) -> None:
    accent = theme["accent"]
    accent_soft = with_alpha(accent, "0.12")
    accent_mid = with_alpha(accent, "0.35")
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {{
  --bg-0: #ffffff;
  --bg-1: #f9f9f9;
  --bg-2: #efefef;
  --bg-3: #dddddd;
  --line: #eeeeee;
  --line-soft: #f5f5f5;
  --fg-0: #000000;
  --fg-1: #111111;
  --fg-2: #555555;
  --fg-3: #888888;
  --accent: {accent};
  --accent-soft: {accent_soft};
  --accent-mid: {accent_mid};
  --brand-red: #e60000;
  --brand-green: #00d95a;
  --r-sm: 2px;
  --r-md: 4px;
  --r-lg: 8px;
  --shadow-card: 0 2px 8px rgba(0,0,0,0.06);
}}

html, body, [class*="stApp"] {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background-color: var(--bg-0);
  color: var(--fg-0);
}}

.stApp {{
  background-color: transparent;
}}

[data-testid="stToolbar"],
[data-testid="stDecoration"],
footer {{
  display: none;
}}

[data-testid="stHeader"] {{
  background: transparent;
}}

.block-container {{
  padding-top: 3rem;
  padding-bottom: 1rem;
  max-width: 1000px;
}}

.nav-brand {{
  margin-bottom: 1.5rem;
}}

.nav-title {{
  font-size: 0.65rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--fg-3);
  margin-bottom: 0.5rem;
}}

.hero-label, .result-label {{
  font-size: 0.65rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  color: var(--brand-red);
  margin-bottom: 0.4rem;
}}

.hero-title {{
  font-size: clamp(1.4rem, 3vw, 2.2rem);
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: -0.02em;
  margin-bottom: 0.75rem;
  color: var(--fg-0);
}}

.hero-title .muted {{
  color: var(--fg-3);
}}

.hero-copy {{
  font-size: 0.9rem;
  line-height: 1.4;
  color: var(--fg-2);
  max-width: 700px;
  margin-bottom: 1.5rem;
}}

.result-container {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1.2rem 0.8rem;
  margin-top: 1.2rem;
  width: 100%;
}}

.result-card {{
  display: flex;
  flex-direction: column;
  text-decoration: none;
  color: inherit;
  transition: opacity 0.2s ease;
  width: 100%;
}}

.result-card:hover {{
  opacity: 0.8;
}}

.album-art {{
  width: 100%;
  aspect-ratio: 1/1;
  background-color: var(--bg-1);
  margin-bottom: 0.6rem;
  position: relative;
  overflow: hidden;
  border-radius: 4px;
}}

.album-art img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.5s ease;
}}

.result-card:hover .album-art img {{
  transform: scale(1.05);
}}

.album-play-overlay {{
  position: absolute;
  bottom: 0.5rem;
  right: 0.5rem;
  width: 1.75rem;
  height: 1.75rem;
  background: #ffcc00;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transform: translateY(8px);
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}}

.album-play-overlay svg {{
  width: 1rem;
  height: 1rem;
  fill: #000;
  margin-left: 1px;
}}

.result-card:hover .album-play-overlay {{
  opacity: 1;
  transform: translateY(0);
}}

.result-info {{
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}}

.result-title {{
  font-size: 0.85rem;
  font-weight: 700;
  line-height: 1.2;
  color: var(--fg-0);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

.result-artist {{
  font-size: 0.75rem;
  color: var(--fg-2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

.result-description {{
  font-size: 0.7rem;
  color: var(--fg-3);
  line-height: 1.3;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  margin-top: 0.2rem;
}}

.mood-pill {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.25rem 0.5rem;
  background: var(--fg-0);
  color: var(--bg-0);
  border-radius: 0;
  font-size: 0.6rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}}

/* Streamlit UI Overrides */
div[data-testid="stForm"] {{
  border: 2px solid var(--fg-0);
  border-radius: 0;
  padding: 0.5rem;
  background: var(--bg-0);
}}

.stTextInput input {{
  border: none !important;
  border-radius: 0 !important;
  font-size: 1.2rem !important;
  padding: 1rem !important;
  font-weight: 600 !important;
}}

div[data-testid="stFormSubmitButton"] > button {{
  background: var(--fg-0) !important;
  color: var(--bg-0) !important;
  border-radius: 0 !important;
  border: none !important;
  font-weight: 800 !important;
  text-transform: uppercase;
  padding: 0 2rem !important;
}}

.stSelectbox div[data-baseweb="select"] {{
  border: 1px solid var(--fg-0) !important;
  border-radius: 0 !important;
}}

.stButton button {{
  border-radius: 0 !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}}
</style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<style>
.stSelectbox svg,
.stSelectbox input { color: var(--fg-1) !important; fill: var(--fg-1) !important; }
.stSlider [data-baseweb="slider"], .stToggle { color: var(--fg-1); }

div[data-testid="stButton"] > button, div[data-testid="stFormSubmitButton"] > button { border-radius: 16px; min-height: 3.05rem; border: 1px solid var(--line-soft) !important; font-weight: 700; transition: all 0.18s ease; }
div[data-testid="stButton"] > button[kind="primary"] { background: linear-gradient(135deg, var(--brand-green), oklch(0.74 0.17 164)) !important; color: oklch(0.12 0.02 180) !important; border: none !important; }
div[data-testid="stFormSubmitButton"] > button[kind="primary"] { background: linear-gradient(135deg, oklch(0.73 0.22 25), oklch(0.68 0.21 18)) !important; color: var(--fg-0) !important; border: none !important; }

.glass-panel, .info-card, .mood-chip-row, .empty-card, .stat-card, .compare-item {
  background: var(--bg-1); border: 1px solid var(--line); padding: 1.5rem; margin-bottom: 1rem;
}
.stat-card { text-align: center; }
.stat-card .value { font-size: 2.5rem; font-weight: 800; color: var(--fg-0); margin-top: 0.5rem; }
.stat-caption { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: var(--fg-3); letter-spacing: 0.1em; }
.stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin: 2rem 0; }
.compare-head { font-size: 1.25rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1.5rem; padding-bottom: 0.5rem; border-bottom: 3px solid var(--fg-0); }
.compare-item { display: flex; flex-direction: column; gap: 0.25rem; }
.compare-rank { font-weight: 800; margin-right: 0.5rem; color: var(--brand-red); }
.compare-table { width: 100%; border-collapse: collapse; margin-top: 2rem; }
.compare-table th { text-align: left; padding: 1rem; background: var(--fg-0); color: var(--bg-0); text-transform: uppercase; font-size: 0.75rem; font-weight: 800; }
.compare-table td { padding: 1rem; border-bottom: 1px solid var(--line); font-size: 0.9rem; }
.matching-shell { padding: 4rem 0; text-align: center; }
.matching-title { font-size: 3rem; font-weight: 800; line-height: 1.1; margin-bottom: 1rem; }
.matching-query { font-size: 1.25rem; color: var(--fg-3); margin-bottom: 3rem; }
.empty-card { text-align: center; padding: 4rem 2rem; }
.empty-card h3 { font-size: 1.5rem; font-weight: 800; margin-bottom: 1rem; }
.mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }
.helper-text { font-size: 0.9rem; color: var(--fg-2); line-height: 1.6; }

@media (max-width: 1100px) {
  .stats-grid { grid-template-columns: 1fr; }
}

/* ── Mobile ─────────────────────────────────── */
@media (max-width: 768px) {
  .block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-top: 3rem !important;
  }
  .result-container {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem 0.6rem;
  }
  .hero-title {
    font-size: 1.8rem !important;
  }
  .matching-title {
    font-size: 1.8rem !important;
  }
  .stats-grid {
    grid-template-columns: 1fr;
  }
  .compare-table th, .compare-table td {
    padding: 0.5rem;
    font-size: 0.75rem;
  }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_mood_pill(theme: dict, score: float) -> str:
    return (
        "<div class='mood-pill'>"
        "<span class='dot'></span>"
        "<span>감지된 무드</span>"
        f"<strong>{escape(theme['label'])}</strong>"
        f"<span class='mono'>{score * 100:.0f}%</span>"
        "</div>"
    )


def render_album_art(song: dict, accent: str) -> str:
    image_url = song.get("image_url") or get_album_art_url(song["title"], song["artist"])
    if image_url:
        return (
            f"<div class='album-art'>"
            f"<img src='{image_url}' alt='{escape(song['title'])}' style='width:100%; height:100%; object-fit:cover;'>"
            f"<div class='album-play-overlay'><svg viewBox='0 0 24 24'><path d='M8 5v14l11-7z'/></svg></div>"
            f"</div>"
        )
    
    initials = escape(song["title"].replace(" ", "")[:2])
    return (
        f"<div class='album-art' style='background: {accent};'>"
        f"<div style='position:absolute; inset:0; display:grid; place-items:center; color:white; font-weight:800; font-size:2rem; opacity:0.3;'>{initials}</div>"
        f"<div class='album-play-overlay'><svg viewBox='0 0 24 24'><path d='M8 5v14l11-7z'/></svg></div>"
        f"</div>"
    )


def render_result_card(song: dict, top_mood: str, accent: str) -> str:
    excerpt = truncate_text(song.get("lyrics", ""), limit=100)
    return f"""
<a class="result-card" href="{escape(song['youtube_music_url'])}" target="_blank">
  {render_album_art(song, accent)}
  <div class="result-info">
    <div class="result-title">{escape(song['title'])}</div>
    <div class="result-artist">{escape(song['artist'])}</div>
    <div class="result-description">{escape(excerpt)}</div>
  </div>
</a>
    """


def queue_search(query: str, source: str = "input") -> None:
    updates = build_search_state_update(query, source=source)
    if not updates:
        st.warning("검색할 감성을 먼저 입력해주세요.")
        return
    for key, value in updates.items():
        st.session_state[key] = value


def execute_search(query: str) -> dict:
    # 1. Spotify 실시간 검색 모드
    if st.session_state.get("use_spotify"):
        with st.spinner("Spotify에서 실시간 추천 곡을 찾는 중..."):
            features = map_mood_to_spotify_features(query)
            sp = SpotifyClient()
            # 국내/국외 설정 전달
            region = st.session_state.get("music_region", "국내")
            spotify_results = sp.search_recommendations(
                features,
                limit=st.session_state["top_k"],
                region=region,
                query=query,
            )
            
            if spotify_results:
                # 무드 분석
                query_vec = embed_text(query)
                client = get_openai_client()
                mood_embeddings = load_mood_embeddings_cache()
                if mood_embeddings is None:
                    mood_embeddings = load_or_create_mood_embeddings(client)
                from classify import classify_mood
                mood_ranking = classify_mood(query_vec, mood_embeddings)
                top_mood, top_mood_score = mood_ranking[0]

                return {
                    "last_query": query,
                    "pending_query": "",
                    "enriched_results": spotify_results,
                    "simple_results": spotify_results,
                    "kw_results": [],
                    "mood_ranking": mood_ranking,
                    "top_mood": top_mood,
                    "top_mood_score": top_mood_score,
                    "query_vec": query_vec,
                    "result_summary": f"Spotify에서 '{query}'에 어울리는 곡들을 실시간으로 가져왔습니다.",
                    "comparison": None,
                    "last_search_settings": {
                        "top_k": st.session_state["top_k"],
                    },
                    "view": "results",
                    "map_requested": False,
                }
            else:
                # SpotifyClient 내부에서 이미 st.error를 띄웠을 것이므로 안내만 출력
                st.info("Spotify 검색 결과를 찾지 못했습니다. 로컬 데이터베이스 검색으로 전환합니다.")

    # 2. 기존 로컬 검색 모드
    query_vec = embed_text(query)
    client = get_openai_client()
    mood_embeddings = load_mood_embeddings_cache()
    if mood_embeddings is None:
        mood_embeddings = load_or_create_mood_embeddings(client)
    from classify import classify_mood

    mood_ranking = classify_mood(query_vec, mood_embeddings)
    top_mood, top_mood_score = mood_ranking[0]
    prioritized_mood = top_mood

    enriched_cache_data = load_embeddings_cache(enriched=True)
    simple_cache_data = load_embeddings_cache(enriched=False)
    enriched_results = []
    simple_results = []

    if enriched_cache_data is not None:
        enriched_results = search_from_query_vector(
            query_vec=query_vec,
            cache=enriched_cache_data,
            top_k=st.session_state["top_k"],
            mood_filter=None,
            prioritized_mood=prioritized_mood,
            randomize=True,
        )
    if simple_cache_data is not None:
        simple_results = search_from_query_vector(
            query_vec=query_vec,
            cache=simple_cache_data,
            top_k=st.session_state["top_k"],
            mood_filter=None,
            prioritized_mood=prioritized_mood,
            randomize=True,
        )

    kw_results = keyword_search(query, top_k=st.session_state["top_k"])
    for index, row in enumerate(kw_results, start=1):
        row["rank"] = index

    semantic_results = enriched_results if st.session_state["use_enriched"] else simple_results
    comparison = compare_search_results(semantic_results, kw_results, query)
    summary = build_result_summary(
        query,
        top_mood,
        top_mood_score,
        semantic_results[0] if semantic_results else None,
    )

    return {
        "last_query": query,
        "pending_query": "",
        "enriched_results": enriched_results,
        "simple_results": simple_results,
        "kw_results": kw_results,
        "mood_ranking": mood_ranking,
        "top_mood": top_mood,
        "top_mood_score": top_mood_score,
        "query_vec": query_vec,
        "result_summary": summary,
        "comparison": comparison,
        "last_search_settings": {
            "top_k": st.session_state["top_k"],
        },
        "view": "results",
        "map_requested": False,
    }


def render_empty_state(title: str, copy: str) -> None:
    st.markdown(
        f"""
<div class="empty-card">
  <h3>{escape(title)}</h3>
  <p>{escape(copy)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_nav() -> None:
    st.markdown(
        """
<div class="nav-brand">
  <div style="font-size:1.5rem;font-weight:900;letter-spacing:-0.05em;text-transform:uppercase;">MoodTune</div>
  <div style="font-size:0.75rem;font-weight:700;color:var(--fg-3);margin-top:0.25rem;letter-spacing:0.05em;">임베딩 기반 음악 검색</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    for item in NAV_ITEMS:
        disabled = item["id"] != "home" and not st.session_state["last_query"]
        clicked = st.button(
            item["label"],
            key=f"nav_{item['id']}",
            use_container_width=True,
            type="primary" if st.session_state["view"] == item["id"] else "secondary",
            disabled=disabled,
        )
        if clicked:
            st.session_state["view"] = item["id"]
            st.rerun()

    st.markdown("<div style='height:3rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-title'>감성 카테고리</div>", unsafe_allow_html=True)
    for mood in get_sidebar_moods():
        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0;">
  <span class="mood-dot" style="background:{mood['accent']};"></span>
  <span style="color:var(--fg-1);font-size:0.9rem;font-weight:600;">{escape(mood['category'])}</span>
</div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state["last_query"]:
        rerun_clicked = st.button(
            "현재 쿼리로 다시 검색",
            key="rerun_current_query",
            use_container_width=True,
        )
        if rerun_clicked:
            queue_search(st.session_state["last_query"], source="preset")
            st.rerun()

def render_home() -> None:
    prefill_updates = consume_query_prefill(st.session_state)
    for key, value in prefill_updates.items():
        st.session_state[key] = value

    theme = current_theme()
    st.markdown(
        f"""
<div style="padding:4rem 0 2rem;">
  <div class="hero-label">SEMANTIC PLAYLIST SEARCH</div>
  <h1 class="hero-title">
    <span class="muted">지금 이 순간,</span><br/>
    당신의 무드에 맞는 노래.
  </h1>
  <div class="hero-copy">
    어떤 느낌인지 적어주세요. MoodTune이 그 느낌에 딱 맞는 10곡을 찾아드립니다.<br/>
    예: "늦은 밤 드라이브", "비 오는 날 카페", "집중하기 좋은 음악"
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("search_form", clear_on_submit=False):
        search_cols = st.columns([5.2, 1.2])
        with search_cols[0]:
            st.text_input(
                "감성 검색",
                key="query_input",
                label_visibility="collapsed",
                placeholder="지금 어떤 기분이신가요?",
            )
        with search_cols[1]:
            submitted = st.form_submit_button("검색", use_container_width=True)

    if submitted:
        queue_search(st.session_state["query_input"], source="input")
        st.rerun()

    setting_cols = st.columns([1, 1, 1, 1])
    with setting_cols[0]:
        st.radio("음악 범위", ["국내", "국외"], key="music_region", horizontal=True)
    with setting_cols[1]:
        st.toggle("풍부한 임베딩", key="use_enriched")
    with setting_cols[2]:
        st.toggle("Spotify 추천", key="use_spotify")

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='nav-title'>이런 검색어는 어때요?</div>",
        unsafe_allow_html=True,
    )

    for i in range(0, len(QUERY_CHIPS), 3):
        row_chips = QUERY_CHIPS[i:i+3]
        cols = st.columns(len(row_chips))
        for col, chip in zip(cols, row_chips):
            with col:
                if st.button(chip, key=f"chip_{chip}", use_container_width=True):
                    queue_search(chip, source="chip")
                    st.rerun()


def render_matching() -> None:
    query = st.session_state["pending_query"] or st.session_state["query_input"]
    st.markdown(
        f"""
<div class="matching-shell">
  <div class="hero-label">SEARCH TRANSITION</div>
  <div class="matching-title">플레이리스트를 구성 중입니다...</div>
  <div class="matching-query">“{escape(query)}”</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    progress = st.progress(0)
    status = st.empty()
    steps = [
        (18, "감성을 벡터로 변환하는 중..."),
        (46, "음악과의 유사도를 분석하는 중..."),
        (74, "최적의 트랙을 선별하는 중..."),
        (100, "플레이리스트 준비 완료."),
    ]
    for value, label in steps[:-1]:
        progress.progress(value)
        status.markdown(f"<div class='helper-text' style='text-align:center;'>{escape(label)}</div>", unsafe_allow_html=True)
        time.sleep(0.28)

    search_state = execute_search(query)
    for key, value in search_state.items():
        st.session_state[key] = value

    progress.progress(steps[-1][0])
    status.markdown(f"<div class='helper-text' style='text-align:center;'>{escape(steps[-1][1])}</div>", unsafe_allow_html=True)
    time.sleep(0.18)
    st.rerun()


def render_results() -> None:
    results = active_results()
    if not results:
        render_empty_state("결과가 없습니다", "홈 화면에서 감성을 검색해주세요.")
        return

    theme = current_theme()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← 다시 검색", key="back_to_home"):
            st.session_state["view"] = "home"
            st.rerun()
            
    # YouTube Music 결과일 경우 첫 번째 곡 플레이어 표시
    # is_yt = any(song.get("source") == "youtube_music" for song in results)
    # if is_yt:
    #     st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    #     top_song = results[0]
    #     video_id = top_song.get("id")
    #     if video_id:
    #         st.markdown(f"<div class='result-label'>FEATURED TRACK</div>", unsafe_allow_html=True)
    #         st.markdown(
    #             f"""
    #             <div style="border-radius: 8px; overflow: hidden; margin-bottom: 1.5rem; border: 1px solid var(--line);">
    #                 <iframe width="100%" height="200" src="https://www.youtube.com/embed/{video_id}?rel=0&modestbranding=1" 
    #                 title="YouTube video player" frameborder="0" 
    #                 allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
    #                 allowfullscreen></iframe>
    #             </div>
    #             """,
    #             unsafe_allow_html=True
    #         )

    # Spotify 결과일 경우 저장 기능 활성화
    is_spotify = any(song.get("source") == "spotify" for song in results)
    if is_spotify:
        with col2:
            with st.expander("🎵 Spotify 플레이리스트로 저장"):
                st.markdown("""
                <div style='font-size: 0.85rem; color: var(--fg-2); margin-bottom: 1rem;'>
                스포티파이 계정에 로그인하여 현재 플레이리스트를 즉시 저장합니다.
                </div>
                """, unsafe_allow_html=True)
                
                token = st.session_state.get("spotify_access_token")
                if not token:
                    auth_url = get_spotify_auth_url()
                    st.link_button("SPOTIFY 계정 연결 및 저장", auth_url, use_container_width=True)
                else:
                    if st.button("플레이리스트 저장", use_container_width=True):
                        try:
                            sp_client = SpotifyClient(user_token=token)
                            track_ids = [s["id"] for s in results if s.get("id")]
                            playlist_name = f"MoodTune: {st.session_state['last_query']}"
                            playlist = sp_client.create_playlist(
                                name=playlist_name,
                                description=f"MoodTune이 추천한 '{st.session_state['last_query']}' 무드 플레이리스트입니다.",
                                track_ids=track_ids
                            )
                            if playlist:
                                st.success(f"'{playlist_name}' 저장 완료!")
                                st.session_state["last_created_playlist_id"] = playlist["id"]
                                st.rerun()
                        except Exception as e:
                            st.error(f"저장 중 오류가 발생했습니다: {e}")
                    if st.button("로그아웃", use_container_width=True):
                        st.session_state["spotify_access_token"] = None
                        st.rerun()

    if st.session_state.get("last_created_playlist_id"):
        pid = st.session_state["last_created_playlist_id"]
        st.markdown(
            f"""
            <iframe src="https://open.spotify.com/embed/playlist/{pid}?utm_source=generator&theme=0" 
            width="100%" height="380" frameBorder="0" allowfullscreen="" 
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='result-label'>PLAYLIST RECOMMENDATIONS</div>", unsafe_allow_html=True)
    st.markdown(
        f"<h1 class='hero-title' style='margin-bottom:0.5rem;'>“{escape(st.session_state['last_query'])}”</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(render_mood_pill(theme, st.session_state["top_mood_score"]), unsafe_allow_html=True)

    cards_html = "".join(
        render_result_card(song, st.session_state["top_mood"], theme["accent"])
        for song in results
    )
    st.markdown(f"<div class='result-container'>{cards_html}</div>", unsafe_allow_html=True)


def render_map() -> None:
    results = active_results()
    if not results or st.session_state["query_vec"] is None:
        render_empty_state("무드 맵을 열 수 없습니다", "먼저 검색을 실행해주세요.")
        return

    theme = current_theme()
    st.markdown("<div class='result-label'>무드 맵</div>", unsafe_allow_html=True)
    st.markdown(
        f"<h1 style='font-size:2.2rem;line-height:1.08;letter-spacing:-0.03em;margin:0.55rem 0 0.45rem;'>“{escape(st.session_state['last_query'])}”의 좌표</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='helper-text'>현재 무드는 <strong style='color:{theme['accent']};'>{escape(theme['label'])}</strong>로 정렬되었습니다. 버튼을 누르면 실제 임베딩 기반 t-SNE 좌표를 계산합니다.</div>",
        unsafe_allow_html=True,
    )

    if st.button("t-SNE 시각화 생성", key="generate_map", type="primary"):
        st.session_state["map_requested"] = True

    if not st.session_state["map_requested"]:
        render_empty_state("시각화 대기 중", "버튼을 누르면 현재 쿼리와 100곡 임베딩을 같은 공간에 배치합니다.")
        return

    with st.spinner("무드 좌표를 계산하는 중..."):
        import matplotlib.pyplot as plt
        from tsne_visualizer import (
            DEFAULT_COLOR,
            MOOD_COLOR_MAP,
            _setup_korean_font,
            get_song_primary_mood,
            load_embeddings,
        )

        _setup_korean_font()
        _, songs_list = load_embeddings(enriched=st.session_state["use_enriched"])
        song_coords_2d, query_coord_2d = get_tsne_coords_for_query(
            st.session_state["query_vec"],
            enriched=st.session_state["use_enriched"],
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f9f9f9")

        plotted = set()
        for song, (x, y) in zip(songs_list, song_coords_2d):
            mood = get_song_primary_mood(song.get("mood_tags", []))
            color = MOOD_COLOR_MAP.get(mood, DEFAULT_COLOR)
            label = mood if mood not in plotted else None
            ax.scatter(
                x,
                y,
                c=color,
                s=100,
                alpha=0.6,
                label=label,
                edgecolors="none",
            )
            plotted.add(mood)

        ax.scatter(
            query_coord_2d[0],
            query_coord_2d[1],
            c="#e60000",
            s=360,
            marker="*",
            zorder=10,
            label=f"★ 당신의 무드",
            edgecolors="black",
            linewidths=1.5,
        )

        ax.set_title("MoodTune Embedding Map", color="#000000", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#888888")
        ax.legend(
            loc="upper right",
            fontsize=8,
            framealpha=0.5,
            facecolor="#ffffff",
            edgecolor="#eeeeee",
            labelcolor="#000000",
        )
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(
            """
<div class="info-card">
  <div class="helper-text" style="font-size: 0.95rem; line-height: 1.6;">
    이 지도는 AI가 파악한 음악들의 <strong>'감성 거리'</strong>를 보여줍니다.<br><br>
    - <strong>색깔 점들</strong>: 서비스에 등록된 다양한 노래들입니다. 끼리끼리 모여있는 점들은 서로 비슷한 분위기를 가진 곡들이에요.<br>
    - <strong>빨간 별표(★)</strong>: 지금 당신이 검색한 기분의 위치입니다.<br>
    - <strong>결과</strong>: 별표 주변에 옹기종기 모여있는 점들이 바로 당신의 무드와 가장 닮았다고 판단되어 추천된 곡들입니다.
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

def render_compare() -> None:
    results = active_results()
    if not results:
        render_empty_state("비교 분석 불가", "검색 결과가 있어야 시맨틱 결과와 키워드 결과를 비교할 수 있습니다.")
        return

    comparison = st.session_state["comparison"] or compare_search_results(
        results,
        st.session_state["kw_results"],
        st.session_state["last_query"],
    )

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='result-label'>COMPARATIVE ANALYSIS</div>", unsafe_allow_html=True)
    st.markdown(
        f"<h1 class='hero-title'>시맨틱 vs 키워드</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-caption">시맨틱 전용</div>
    <div class="value">{len(comparison['semantic_only'])}</div>
  </div>
  <div class="stat-card">
    <div class="stat-caption">키워드 전용</div>
    <div class="value">{len(comparison['keyword_only'])}</div>
  </div>
  <div class="stat-card">
    <div class="stat-caption">중복 결과</div>
    <div class="value">{comparison['overlap_count']}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    semantic_col, keyword_col = st.columns(2, gap="large")

    with semantic_col:
        st.markdown("<div class='compare-head'>Semantic Results</div>", unsafe_allow_html=True)
        semantic_items = []
        for idx, row in enumerate(results, start=1):
            rank = row.get("rank", idx)
            similarity = row.get("similarity", 0.0)
            semantic_items.append(
                f"""
<div class="compare-item">
  <div><span class="compare-rank">{rank:02d}</span><strong>{escape(row['title'])}</strong> <span class="result-artist">- {escape(row['artist'])}</span></div>
  <div class="helper-text">유사도 {similarity:.4f}</div>
</div>
                """
            )
        st.markdown("<div class='compare-list'>" + "".join(semantic_items) + "</div>", unsafe_allow_html=True)

    with keyword_col:
        st.markdown("<div class='compare-head'>Keyword Results</div>", unsafe_allow_html=True)
        keyword_items = []
        for row in st.session_state["kw_results"]:
            common_keywords = ", ".join(row.get("common_keywords", [])[:4]) or "없음"
            keyword_items.append(
                f"""
<div class="compare-item">
  <div><span class="compare-rank">{row['rank']:02d}</span><strong>{escape(row['title'])}</strong> <span class="result-artist">- {escape(row['artist'])}</span></div>
  <div class="helper-text">TF-IDF {row['tfidf_similarity']:.4f} · 키워드: {escape(common_keywords)}</div>
</div>
            """
            )
        if not keyword_items:
            keyword_items.append("<div class='helper-text'>키워드 검색 결과 없음 (Spotify 모드)</div>")
        st.markdown("<div class='compare-list'>" + "".join(keyword_items) + "</div>", unsafe_allow_html=True)

    compare_rows = build_compare_rows(
        st.session_state.get("simple_results", []),
        st.session_state.get("enriched_results", []),
    )
    body = []
    for row in compare_rows:
        simple = row["simple"]
        enriched = row["enriched"]
        simple_text = "없음"
        if simple is not None:
            simple_text = f"{simple['title']} - {simple['artist']} ({simple['similarity']:.4f})"
        enriched_text = "없음"
        if enriched is not None:
            enriched_text = f"{enriched['title']} - {enriched['artist']} ({enriched['similarity']:.4f})"
        body.append(
            f"""
<tr>
  <td class="mono">{row['rank']:02d}</td>
  <td>{escape(simple_text)}</td>
  <td>{escape(enriched_text)}</td>
</tr>
            """
        )

    st.markdown(
        """
<div style="height:3rem;"></div>
<div class="compare-head">단순 임베딩 vs 풍부한 임베딩</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<table class='compare-table'><thead><tr><th>Rank</th><th>Simple</th><th>Enriched</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>",
        unsafe_allow_html=True,
    )


def render_main() -> None:
    view = st.session_state["view"]
    if view == "home":
        render_home()
    elif view == "matching":
        render_matching()
    elif view == "results":
        render_results()
    elif view == "map":
        render_map()
    elif view == "compare":
        render_compare()
    else:
        render_home()


def handle_spotify_oauth_callback() -> None:
    """URL의 ?code= 파라미터를 감지하여 액세스 토큰으로 교환"""
    params = st.query_params
    code = params.get("code")
    if code and not st.session_state.get("spotify_access_token"):
        token_info = exchange_code_for_token(code)
        if token_info:
            st.session_state["spotify_access_token"] = token_info["access_token"]
        st.query_params.clear()
        st.rerun()


def main() -> None:
    init_state()
    handle_spotify_oauth_callback()
    apply_theme(current_theme())

    enriched_cache_data = load_embeddings_cache(enriched=True)
    simple_cache_data = load_embeddings_cache(enriched=False)
    if enriched_cache_data is None and simple_cache_data is None:
        render_empty_state(
            "임베딩 캐시가 없습니다",
            "먼저 'python embed_songs.py'를 실행하여 embeddings.pkl과 enriched_embeddings.pkl을 생성해주세요.",
        )
        st.stop()

    with st.sidebar:
        render_nav()
    render_main()


main()
