from collections import Counter
from html import escape

from .benchmark import flatten_benchmark_summary
from .comparison import build_backend_comparison_rows
from .data import load_song_dataset
from .filters import build_metadata_filter
from .services import (
    build_vector_map,
    compare_all_backends,
    delete_backend_records,
    get_backend_record,
    query_backend,
    run_benchmark,
    run_chroma_update_vs_upsert_demo,
    run_hybrid_demo,
    run_pinecone_idempotency_demo,
    sync_backend,
)


def available_genres() -> list[str]:
    songs = load_song_dataset()
    genres = sorted({song.get("genre", "unknown") for song in songs})
    return ["전체"] + genres


def available_moods() -> list[str]:
    songs = load_song_dataset()
    counter = Counter()
    for song in songs:
        for mood in song.get("mood_tags", []):
            counter[mood] += 1
    return ["전체"] + [mood for mood, _ in counter.most_common()]


def build_backend_filter(
    backend: str,
    genre: str = "전체",
    primary_mood: str = "전체",
    min_lyrics_length: int = 0,
    exclude_genre: str = "없음",
) -> dict | None:
    return build_metadata_filter(
        genre=genre,
        primary_mood=primary_mood,
        min_lyrics_length=min_lyrics_length,
        exclude_genre=exclude_genre,
    )


def build_vector_result_markup(row: dict) -> str:
    score = row.get("score", 0.0)
    distance = row.get("distance")
    distance_text = f"{distance:.4f}" if isinstance(distance, (int, float)) else "-"
    return (
        "<div style='border:1px solid rgba(0,0,0,0.12); padding:1rem; margin-bottom:0.85rem; "
        "border-radius:18px; background:white;'>"
        f"<div style='font-weight:800; font-size:1.05rem;'>{escape(str(row.get('title', row.get('song_id', row.get('id', '')))))}</div>"
        f"<div style='color:#666; margin:0.2rem 0 0.5rem;'>{escape(str(row.get('artist', '')))}</div>"
        "<div style='display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.55rem;'>"
        f"<span style='padding:0.2rem 0.55rem; border-radius:999px; background:#eef4ff;'>genre {escape(str(row.get('genre', '-')))}</span>"
        f"<span style='padding:0.2rem 0.55rem; border-radius:999px; background:#effcf2;'>mood {escape(str(row.get('primary_mood', '-')))}</span>"
        f"<span style='padding:0.2rem 0.55rem; border-radius:999px; background:#fff5eb;'>lyrics {escape(str(row.get('lyrics_length', '-')))}</span>"
        "</div>"
        f"<div style='font-family:ui-monospace, monospace; font-size:0.92rem;'>score {score:.4f} | distance {distance_text}</div>"
        "</div>"
    )


def render_vector_db_page() -> None:
    import streamlit as st

    st.markdown("## Vector DB Lab")
    st.markdown("ChromaDB와 Pinecone을 같은 데이터셋으로 비교하는 과제용 화면입니다.")

    control_cols = st.columns([1.1, 1.1, 1.1, 1.1])
    with control_cols[0]:
        backend = st.selectbox("백엔드", ["chroma", "pinecone"], key="vector_backend")
    with control_cols[1]:
        genre = st.selectbox("장르 필터", available_genres(), key="vector_genre")
    with control_cols[2]:
        primary_mood = st.selectbox("무드 필터", available_moods(), key="vector_primary_mood")
    with control_cols[3]:
        exclude_genre = st.selectbox("제외 장르", ["없음"] + available_genres()[1:], key="vector_exclude_genre")

    lower_cols = st.columns([1.3, 1, 1, 1])
    with lower_cols[0]:
        query = st.text_input("Vector DB 질의", value=st.session_state.get("last_query", "카페에서 집중할 때 듣는 노래"), key="vector_query")
    with lower_cols[1]:
        top_k = st.slider("Top K", min_value=3, max_value=10, value=5, key="vector_top_k")
    with lower_cols[2]:
        min_lyrics_length = st.slider("최소 가사 길이", min_value=0, max_value=400, value=80, step=20, key="vector_min_lyrics_length")
    with lower_cols[3]:
        scope = st.selectbox("컬렉션/네임스페이스", ["all"], key="vector_scope")

    filter_expr = build_backend_filter(
        backend=backend,
        genre=genre,
        primary_mood=primary_mood,
        min_lyrics_length=min_lyrics_length,
        exclude_genre=exclude_genre,
    )
    st.code(str(filter_expr or {}), language="python")

    action_cols = st.columns(4)
    with action_cols[0]:
        sync_clicked = st.button("동기화", use_container_width=True, key="vector_sync")
    with action_cols[1]:
        search_clicked = st.button("검색", use_container_width=True, key="vector_search")
    with action_cols[2]:
        compare_clicked = st.button("3-way 비교", use_container_width=True, key="vector_compare")
    with action_cols[3]:
        hybrid_clicked = st.button("하이브리드 데모", use_container_width=True, key="vector_hybrid")

    crud_cols = st.columns(2)
    with crud_cols[0]:
        lookup_id = st.text_input("ID 조회", value="song_001", key="vector_lookup_id")
    with crud_cols[1]:
        delete_id = st.text_input("삭제 ID", value="demo_missing_record", key="vector_delete_id")

    crud_action_cols = st.columns(2)
    with crud_action_cols[0]:
        get_clicked = st.button("ID 조회 실행", use_container_width=True, key="vector_get")
    with crud_action_cols[1]:
        delete_clicked = st.button("ID 삭제 실행", use_container_width=True, key="vector_delete")

    if sync_clicked:
        with st.spinner(f"{backend} 동기화 중..."):
            counts = sync_backend(backend)
        st.session_state["vector_sync_counts"] = counts

    if st.session_state.get("vector_sync_counts") is not None:
        st.json(st.session_state["vector_sync_counts"])

    if get_clicked:
        with st.spinner(f"{backend} ID 조회 중..."):
            st.session_state["vector_get_payload"] = get_backend_record(backend=backend, record_id=lookup_id, scope=scope)

    if delete_clicked:
        with st.spinner(f"{backend} 삭제 중..."):
            st.session_state["vector_delete_payload"] = delete_backend_records(
                backend=backend,
                record_id=delete_id,
                scope=scope,
            )

    if search_clicked:
        with st.spinner(f"{backend} 검색 중..."):
            rows = query_backend(
                backend=backend,
                query=query,
                top_k=top_k,
                scope=scope,
                metadata_filter=filter_expr,
            )
        st.session_state["vector_rows"] = rows

    if hybrid_clicked:
        with st.spinner(f"{backend} 하이브리드 시나리오 실행 중..."):
            st.session_state["vector_hybrid_rows"] = run_hybrid_demo(backend, top_k=top_k)

    if compare_clicked:
        with st.spinner("파일 기반, ChromaDB, Pinecone 비교 중..."):
            st.session_state["vector_compare_payload"] = compare_all_backends(query, top_k=top_k)

    rows = st.session_state.get("vector_rows", [])
    if rows:
        st.markdown("### Search Results")
        st.markdown("".join(build_vector_result_markup(row) for row in rows), unsafe_allow_html=True)

    get_payload = st.session_state.get("vector_get_payload")
    if get_payload:
        st.markdown("### Get By ID")
        st.json(get_payload)

    delete_payload = st.session_state.get("vector_delete_payload")
    if delete_payload:
        st.markdown("### Delete Result")
        st.json(delete_payload)

    hybrid_rows = st.session_state.get("vector_hybrid_rows", [])
    if hybrid_rows:
        st.markdown("### Hybrid Search Scenarios")
        for row in hybrid_rows:
            st.markdown(f"#### {row['name']}")
            st.write({"query": row["query"], "filter": row["filter"]})
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Pure Semantic")
                st.markdown("".join(build_vector_result_markup(item) for item in row["pure"]), unsafe_allow_html=True)
            with col2:
                st.markdown("Hybrid Search")
                st.markdown("".join(build_vector_result_markup(item) for item in row["hybrid"]), unsafe_allow_html=True)

    compare_payload = st.session_state.get("vector_compare_payload")
    if compare_payload:
        st.markdown("### Backend Comparison")
        compare_cols = st.columns(3)
        for col, backend_name in zip(compare_cols, ["file", "chroma", "pinecone"]):
            with col:
                st.markdown(f"#### {backend_name}")
                st.markdown(
                    "".join(build_vector_result_markup(item) for item in compare_payload.get(backend_name, [])),
                    unsafe_allow_html=True,
                )

    demo_cols = st.columns(2)
    with demo_cols[0]:
        if st.button("Chroma update vs upsert", use_container_width=True, key="vector_chroma_demo"):
            with st.spinner("Chroma 데모 실행 중..."):
                st.session_state["chroma_demo_payload"] = run_chroma_update_vs_upsert_demo()
    with demo_cols[1]:
        if st.button("Pinecone idempotent upsert", use_container_width=True, key="vector_pinecone_demo"):
            with st.spinner("Pinecone 데모 실행 중..."):
                st.session_state["pinecone_demo_payload"] = run_pinecone_idempotency_demo()

    if st.session_state.get("chroma_demo_payload"):
        st.markdown("### Chroma update vs upsert")
        st.json(st.session_state["chroma_demo_payload"])
    if st.session_state.get("pinecone_demo_payload"):
        st.markdown("### Pinecone idempotent upsert")
        st.json(st.session_state["pinecone_demo_payload"])


def render_vector_map_page() -> None:
    import streamlit as st

    st.markdown("## Vector DB Map")
    top_cols = st.columns(4)
    with top_cols[0]:
        backend = st.selectbox("백엔드", ["chroma", "pinecone"], key="map_backend")
    with top_cols[1]:
        method = st.selectbox("차원 축소", ["tsne", "umap"], key="map_method")
    with top_cols[2]:
        color_by = st.selectbox("색상 기준", ["genre", "primary_mood"], key="map_color_by")
    with top_cols[3]:
        scope = st.selectbox("범위", ["all"], key="map_scope")

    if st.button("DB 맵 생성", type="primary", use_container_width=True, key="build_vector_map"):
        with st.spinner(f"{backend} {method} 시각화 생성 중..."):
            path, _ = build_vector_map(backend=backend, method=method, scope=scope, color_by=color_by)
        st.session_state["vector_map_path"] = path

    path = st.session_state.get("vector_map_path")
    if path:
        st.image(path, use_container_width=True)


def render_vector_benchmark_page() -> None:
    import streamlit as st

    st.markdown("## Benchmark")
    query = st.text_input("벤치마크 질의", value="비 오는 날 카페에서 듣는 노래", key="benchmark_query")
    runs = st.slider("반복 횟수", min_value=1, max_value=5, value=3, key="benchmark_runs")
    if st.button("100 / 500 / 1000 벤치마크 실행", type="primary", use_container_width=True, key="benchmark_run"):
        with st.spinner("벤치마크 실행 중..."):
            rows, summary, plot_path = run_benchmark(query, runs=runs)
        st.session_state["benchmark_rows"] = rows
        st.session_state["benchmark_summary"] = summary
        st.session_state["benchmark_plot_path"] = plot_path

    summary = st.session_state.get("benchmark_summary")
    plot_path = st.session_state.get("benchmark_plot_path")
    st.markdown("### Capability Comparison")
    st.dataframe(build_backend_comparison_rows(), use_container_width=True, hide_index=True)
    if summary:
        st.json(flatten_benchmark_summary(summary))
    if plot_path:
        st.image(plot_path, use_container_width=True)
