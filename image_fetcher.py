import requests
from functools import lru_cache
import urllib.parse

@lru_cache(maxsize=1000)
def get_album_art_url(title: str, artist: str) -> str:
    """
    iTunes Search API를 사용하여 곡의 앨범 아트 URL을 가져옵니다.
    """
    query = f"{title} {artist}"
    encoded_query = urllib.parse.quote(query)
    url = f"https://itunes.apple.com/search?term={encoded_query}&entity=song&limit=1"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data["resultCount"] > 0:
            # 600x600 고해상도 이미지를 위해 URL 수정 (100x100 -> 600x600)
            artwork_url = data["results"][0]["artworkUrl100"]
            return artwork_url.replace("100x100bb.jpg", "600x600bb.jpg")
    except Exception as e:
        print(f"Error fetching album art for {query}: {e}")
    
    return ""
