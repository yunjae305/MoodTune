import os
import requests
import base64
from typing import List, Dict, Optional

class SpotifyClient:
    def __init__(self, client_id: str = None, client_secret: str = None, user_token: str = None):
        self.client_id = client_id or os.environ.get("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")
        self.access_token = user_token # 유저 토큰이 제공되면 우선 사용
        self.user_token_mode = user_token is not None

    def _get_access_token(self) -> bool:
        """Client Credentials Flow를 통해 액세스 토큰을 가져옵니다 (검색용)."""
        if self.user_token_mode:
            return True # 이미 유저 토큰이 있음
            
        if not self.client_id or not self.client_secret:
            return False

        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {auth_header}"}
        )
        
        if response.status_code == 200:
            self.access_token = response.json().get("access_token")
            return True
        return False

    def create_playlist(self, name: str, description: str, track_ids: List[str]) -> Optional[Dict]:
        """사용자 계정에 플레이리스트를 생성하고 곡을 추가합니다."""
        if not self.access_token:
            return None

        headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
        
        # 1. 플레이리스트 생성
        user_res = requests.get("https://api.spotify.com/v1/me", headers=headers)
        if user_res.status_code != 200:
            return None
        
        user_id = user_res.json()["id"]
        create_res = requests.post(
            f"https://api.spotify.com/v1/users/{user_id}/playlists",
            headers=headers,
            json={"name": name, "description": description, "public": False}
        )
        
        if create_res.status_code not in [200, 201]:
            return None
            
        playlist = create_res.json()
        
        # 2. 곡 추가
        track_uris = [f"spotify:track:{tid}" for tid in track_ids]
        add_res = requests.post(
            f"https://api.spotify.com/v1/playlists/{playlist['id']}/items",
            headers=headers,
            json={"uris": track_uris}
        )
        
        if add_res.status_code in [200, 201]:
            return playlist
        return None

    def search_recommendations(self, features: Dict, limit: int = 8) -> List[Dict]:
        """Spotify Recommendations API를 사용하여 곡을 추천받습니다."""
        if not self.access_token and not self._get_access_token():
            return []

        # 추천 API 파라미터 구성
        params = {
            "limit": limit,
            "market": "KR",
            "seed_genres": features.get("seed_genres", "pop,k-pop"),
            "target_energy": features.get("target_energy", 0.5),
            "target_valence": features.get("target_valence", 0.5),
            "target_danceability": features.get("target_danceability", 0.5),
            "target_tempo": features.get("target_tempo", 120),
        }

        response = requests.get(
            "https://api.spotify.com/v1/recommendations",
            params=params,
            headers={"Authorization": f"Bearer {self.access_token}"}
        )

        if response.status_code == 200:
            tracks = response.json().get("tracks", [])
            return [self._format_track(track) for track in tracks]
        return []

    def _format_track(self, track: Dict) -> Dict:
        """Spotify 트랙 정보를 MoodTune 형식으로 변환합니다."""
        return {
            "id": track["id"],
            "title": track["name"],
            "artist": ", ".join([a["name"] for a in track["artists"]]),
            "album": track["album"]["name"],
            "image_url": track["album"]["images"][0]["url"] if track["album"]["images"] else "",
            "youtube_music_url": f"https://music.youtube.com/search?q={track['name']} {track['artists'][0]['name']}",
            "spotify_url": track["external_urls"]["spotify"],
            "preview_url": track["preview_url"],
            "source": "spotify"
        }
