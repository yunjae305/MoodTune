import os
from typing import Dict, List, Optional

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth


load_dotenv(override=True)

SPOTIFY_SCOPE = "playlist-modify-private playlist-modify-public"


def normalize_seed_genres(raw_genres) -> List[str]:
    if isinstance(raw_genres, list):
        values = raw_genres
    else:
        values = str(raw_genres or "").split(",")
    genres = []
    seen = set()
    for value in values:
        genre = str(value).strip()
        if not genre:
            continue
        lowered = genre.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        genres.append(lowered)
    return genres


def build_query_hints(query: str) -> List[str]:
    text = (query or "").strip().lower()
    hints = []
    keyword_groups = [
        (["비", "rain"], "rainy"),
        (["밤", "새벽", "night", "midnight", "late"], "night"),
        (["잔잔", "조용", "calm", "quiet"], "calm"),
        (["몽환", "dream", "dreamy"], "dreamy"),
        (["드라이브", "drive"], "drive"),
        (["운동", "헬스", "workout", "gym"], "workout"),
        (["신나", "에너지", "party", "dance"], "energetic"),
        (["우울", "슬픔", "sad"], "sad"),
    ]
    for keywords, label in keyword_groups:
        if any(keyword in text for keyword in keywords):
            hints.append(label)
    return hints


def build_search_queries(query: str, seed_genres: List[str], region: str, features: Dict) -> List[str]:
    base_query = (query or "").strip()
    genre_terms = " ".join(seed_genres[:2]).strip()
    raw_search_terms = features.get("search_terms", [])
    if isinstance(raw_search_terms, list):
        search_terms = [str(item).strip() for item in raw_search_terms if str(item).strip()]
    else:
        search_terms = [str(raw_search_terms).strip()] if str(raw_search_terms).strip() else []
    query_hints = build_query_hints(base_query)
    descriptors = []
    energy = float(features.get("target_energy", 0.5))
    valence = float(features.get("target_valence", 0.5))
    danceability = float(features.get("target_danceability", 0.5))
    tempo = float(features.get("target_tempo", 110))

    if energy <= 0.35:
        descriptors.append("calm")
    elif energy >= 0.75:
        descriptors.append("energetic")

    if valence <= 0.4:
        descriptors.append("moody")
    elif valence >= 0.65:
        descriptors.append("bright")

    if danceability <= 0.35:
        descriptors.append("dreamy")
    elif danceability >= 0.7:
        descriptors.append("dance")

    if tempo >= 140:
        descriptors.append("fast")
    elif tempo <= 85:
        descriptors.append("slow")

    region_terms = ["k-pop", "korean"] if region == "국내" else ["global pop"]
    candidates = [
        base_query,
        " ".join(part for part in [base_query, " ".join(query_hints[:2])] if part).strip(),
        " ".join(part for part in [base_query, " ".join(search_terms[:2])] if part).strip(),
        " ".join(part for part in [base_query, genre_terms] if part).strip(),
        " ".join(part for part in [base_query, " ".join(descriptors[:2])] if part).strip(),
        " ".join(part for part in [genre_terms, " ".join(query_hints[:2])] if part).strip(),
        " ".join(part for part in [genre_terms, " ".join(search_terms[:2])] if part).strip(),
        " ".join(part for part in [genre_terms, " ".join(descriptors[:2]), region_terms[0]] if part).strip(),
        " ".join(part for part in [base_query, region_terms[0]] if part).strip(),
        " ".join(part for part in [genre_terms, region_terms[-1]] if part).strip(),
    ]

    queries = []
    seen = set()
    for candidate in candidates:
        value = candidate.strip()
        if not value:
            continue
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        queries.append(value)

    if not queries:
        queries.append("k-pop" if region == "국내" else "pop")

    return queries


def rank_search_candidates(candidates: List[tuple]) -> List[Dict]:
    total_queries = max([query_index for query_index, _, _ in candidates], default=-1) + 1
    ranked = {}
    for query_index, item_index, track in candidates:
        track_id = track.get("id")
        if not track_id:
            continue
        popularity = float(track.get("popularity", 0.0))
        score = (total_queries - query_index) * 20 + max(0, 20 - item_index * 2) + popularity
        saved = ranked.get(track_id)
        if saved is None:
            ranked[track_id] = {"score": score, "track": track}
            continue
        saved["score"] += score

    ordered = sorted(ranked.values(), key=lambda item: item["score"], reverse=True)
    return [item["track"] for item in ordered]


def get_oauth_manager() -> SpotifyOAuth:
    return SpotifyOAuth(
        client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
        client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.environ.get("SPOTIFY_REDIRECT_URI", "http://localhost:8501/"),
        scope=SPOTIFY_SCOPE,
        open_browser=False,
        cache_handler=None,
    )


def get_spotify_auth_url() -> str:
    return get_oauth_manager().get_authorize_url()


def exchange_code_for_token(code: str) -> Optional[Dict]:
    try:
        return get_oauth_manager().get_access_token(code, as_dict=True, check_cache=False)
    except Exception as e:
        print(f"토큰 교환 실패: {e}")
        return None


class SpotifyClient:
    def __init__(self, client_id: str = None, client_secret: str = None, user_token: str = None):
        self.client_id = client_id or os.environ.get("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
        self.user_token = user_token

        try:
            if user_token:
                self.sp = spotipy.Spotify(auth=user_token)
            else:
                auth_manager = SpotifyClientCredentials(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                )
                self.sp = spotipy.Spotify(auth_manager=auth_manager)
        except Exception as e:
            print(f"Spotify 연결 실패: {e}")
            self.sp = None

    def get_available_genres(self) -> List[str]:
        return []

    def search_recommendations(self, features: Dict, limit: int = 8, region: str = "국내", query: str = "") -> List[Dict]:
        if not self.sp:
            import streamlit as st

            st.error("Spotify 설정을 완료해주세요 (ID/Secret 필요)")
            return []

        try:
            seed_genres = normalize_seed_genres(features.get("seed_genres", ""))
            market = "KR" if region == "국내" else "US"
            query_variants = build_search_queries(
                query=query,
                seed_genres=seed_genres,
                region=region,
                features=features,
            )
            candidates = []
            per_query_limit = max(limit, 6)

            for query_index, search_query in enumerate(query_variants):
                search_results = self.sp.search(
                    q=search_query,
                    type="track",
                    limit=per_query_limit,
                    market=market,
                )
                tracks = search_results.get("tracks", {}).get("items", [])
                for item_index, track in enumerate(tracks):
                    candidates.append((query_index, item_index, track))

            ranked_tracks = rank_search_candidates(candidates)
            return [self._format_track(track) for track in ranked_tracks[:limit]]
        except Exception as e:
            import streamlit as st

            st.error(f"[Spotify 에러] 연동 실패: {type(e).__name__}: {e}")
            return []

    def create_playlist(self, name: str, description: str, track_ids: List[str]) -> Optional[Dict]:
        if not self.user_token or not self.sp:
            return None
        try:
            user_id = self.sp.current_user()["id"]
            playlist = self.sp.user_playlist_create(user_id, name, public=False, description=description)
            self.sp.playlist_add_items(playlist["id"], track_ids)
            return playlist
        except Exception as e:
            import streamlit as st

            st.error(f"플레이리스트 생성 실패: {e}")
            return None

    def _format_track(self, track: Dict) -> Dict:
        spotify_url = track["external_urls"]["spotify"]
        return {
            "id": track["id"],
            "title": track["name"],
            "artist": ", ".join([artist["name"] for artist in track["artists"]]),
            "album": track["album"]["name"],
            "image_url": track["album"]["images"][0]["url"] if track["album"]["images"] else "",
            "youtube_music_url": spotify_url,
            "spotify_url": spotify_url,
            "preview_url": track.get("preview_url"),
            "source": "spotify",
        }
