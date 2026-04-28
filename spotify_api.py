import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from typing import List, Dict, Optional
from dotenv import load_dotenv

# 환경 변수 강제 로드
load_dotenv(override=True)

SPOTIFY_SCOPE = "playlist-modify-private playlist-modify-public"


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
    """사용자를 Spotify 로그인 페이지로 보내기 위한 URL 반환"""
    return get_oauth_manager().get_authorize_url()


def exchange_code_for_token(code: str) -> Optional[Dict]:
    """Spotify가 리다이렉트로 돌려준 code를 access_token으로 교환"""
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
                # Client Credentials (단순 검색용)
                auth_manager = SpotifyClientCredentials(
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
                self.sp = spotipy.Spotify(auth_manager=auth_manager)
        except Exception as e:
            print(f"Spotify 연결 실패: {e}")
            self.sp = None

    def get_available_genres(self) -> List[str]:
        if not self.sp: return []
        try:
            return self.sp.recommendation_genre_seeds()["genres"]
        except:
            return []

    def search_recommendations(self, features: Dict, limit: int = 8, region: str = "국내", query: str = "") -> List[Dict]:
        if not self.sp:
            import streamlit as st
            st.error("Spotify 설정을 완료해주세요 (ID/Secret 미로드)")
            return []

        try:
            # 장르 보정
            available = self.get_available_genres()
            
            # seed_genres가 리스트인 경우 콤마로 연결된 문자열로 변환
            raw_genres = features.get("seed_genres", "pop")
            if isinstance(raw_genres, list):
                raw_genres = ",".join(raw_genres)
            
            valid_genres = [g.strip() for g in raw_genres.split(",") if g.strip() in available]
            if not valid_genres: valid_genres = ["pop"]

            # 지역 설정에 따른 마켓 및 검색어 보정
            market = "KR" if region == "국내" else "US"
            
            # 1. 추천 호출 시도
            try:
                # 국내 음악 위주인 경우 k-pop 장르 강제 추가 시도
                seeds = valid_genres[:5]
                if region == "국내" and "k-pop" in available and "k-pop" not in seeds:
                    seeds = ["k-pop"] + seeds[:4]

                results = self.sp.recommendations(
                    seed_genres=seeds,
                    limit=limit,
                    country=market,
                    target_energy=float(features.get("target_energy", 0.5)),
                    target_valence=float(features.get("target_valence", 0.5)),
                    target_danceability=float(features.get("target_danceability", 0.5))
                )
                return [self._format_track(track) for track in results["tracks"]]
            except Exception as rec_err:
                print(f"Spotify Recommendations API 실패 (market={market}): {rec_err}")
                # 2. 추천 API 실패 시 검색(Search) API로 폴백
                import random as _random
                genre_hint = valid_genres[0] if valid_genres else "pop"

                # 연도 범위를 랜덤으로 선택하여 매번 다른 결과 유도
                year_pools = [
                    "2020-2025", "2018-2022", "2015-2020",
                    "2012-2018", "2010-2015", "2005-2012",
                ]
                year_filter = _random.choice(year_pools)

                # 랜덤 offset (0~40 사이, 10 단위)
                offset = _random.choice([0, 10, 20, 30, 40])

                if query:
                    base = f"{query} {genre_hint} K-pop" if region == "국내" else f"{query} {genre_hint}"
                elif region == "국내":
                    base = f"{genre_hint} K-pop"
                else:
                    base = genre_hint
                search_query = f"{base} year:{year_filter}"

                search_results = self.sp.search(
                    q=search_query, type="track",
                    limit=limit * 2, offset=offset, market=market,
                )
                tracks = search_results["tracks"]["items"]
                _random.shuffle(tracks)
                return [self._format_track(t) for t in tracks[:limit]]

        except Exception as e:
            import streamlit as st
            st.error(f"[Spotify 에러] 연동 실패: {e}")
            return []

    def create_playlist(self, name: str, description: str, track_ids: List[str]) -> Optional[Dict]:
        if not self.user_token or not self.sp: return None
        try:
            user_id = self.sp.current_user()["id"]
            playlist = self.sp.user_playlist_create(user_id, name, public=False, description=description)
            self.sp.playlist_add_items(playlist["id"], track_ids)
            return playlist
        except Exception as e:
            import streamlit as st
            st.error(f"플레이리스트 저장 실패: {e}")
            return None

    def _format_track(self, track: Dict) -> Dict:
        return {
            "id": track["id"],
            "title": track["name"],
            "artist": ", ".join([a["name"] for a in track["artists"]]),
            "album": track["album"]["name"],
            "image_url": track["album"]["images"][0]["url"] if track["album"]["images"] else "",
            "youtube_music_url": f"https://music.youtube.com/watch?v={track['name']} {track['artists'][0]['name']}",
            "spotify_url": track["external_urls"]["spotify"],
            "preview_url": track.get("preview_url"),
            "source": "spotify"
        }
