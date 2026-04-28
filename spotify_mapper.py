import json
from openai import OpenAI
import os

def map_mood_to_spotify_features(query: str) -> dict:
    """
    사용자의 자연어 쿼리를 Spotify API에서 사용할 수 있는 오디오 특징 파라미터로 변환합니다.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    prompt = f"""
    당신은 음악 전문가입니다. 사용자의 기분이나 상황을 나타내는 문장을 듣고, 이에 어울리는 음악을 Spotify에서 찾기 위한 오디오 특징 값을 JSON 형식으로 추천해야 합니다.

    입력 문장: "{query}"

    다음 필드들을 포함하는 JSON을 반환하세요:
    1. seed_genres: 해당 무드에 어울리는 Spotify 장르 1~2개. 
       반드시 다음 허용된 목록 중에서만 선택하세요: [acoustic, afrobeat, alt-rock, alternative, ambient, anime, black-metal, bluegrass, blues, bossanova, brazil, breakbeat, british, cantopop, chicago-house, children, chill, classical, club, comedy, country, dance, dancehall, death-metal, deep-house, detroit-techno, disco, disney, drum-and-bass, dub, dubstep, edm, electro, electronic, emo, folk, forro, french, funk, garage, german, gospel, goth, grindcore, groove, grunge, guitar, happy, hard-rock, hardcore, hardstyle, heavy-metal, hip-hop, holidays, honky-tonk, house, idm, indian, indie, indie-pop, industrial, Iranian, j-dance, j-idol, j-pop, j-rock, jazz, k-pop, kids, latin, latino, malay, mandopop, metal, metal-core, metal-heavy, minimal-techno, movies, mpb, new-age, new-release, opera, pagode, party, philippines-opm, piano, pop, pop-film, post-dubstep, power-pop, progressive-house, psych-rock, punk, punk-rock, r-n-b, rainy-day, reggae, reggaeton, road-trip, rock, rock-n-roll, rockabilly, romance, sad, salsa, samba, sertanejo, show-tunes, singer-songwriter, ska, sleep, songwriter, soul, soundtracks, spanish, study, summer, swedish, synth-pop, tango, techno, trance, trip-hop, turkish, work-out, world-music]
       (반드시 소문자 콤마로 구분)
    2. target_energy: 0.0 ~ 1.0 (에너지 레벨)
    3. target_valence: 0.0 ~ 1.0 (음악의 밝기/긍정적 느낌)
    4. target_danceability: 0.0 ~ 1.0 (댄스 지수)
    5. target_tempo: 60 ~ 180 (BPM)

    JSON 형식으로만 대답하세요.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error mapping mood: {e}")
        # 기본값 반환
        return {
            "seed_genres": "pop,k-pop",
            "target_energy": 0.5,
            "target_valence": 0.5,
            "target_danceability": 0.5,
            "target_tempo": 120
        }
