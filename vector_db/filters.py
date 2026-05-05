def build_metadata_filter(
    genre: str = "전체",
    primary_mood: str = "전체",
    min_lyrics_length: int = 0,
    exclude_genre: str = "없음",
) -> dict | None:
    clauses = []
    if genre != "전체":
        clauses.append({"genre": {"$eq": genre}})
    if primary_mood != "전체":
        clauses.append({"primary_mood": {"$eq": primary_mood}})
    if min_lyrics_length > 0:
        clauses.append({"lyrics_length": {"$gte": min_lyrics_length}})
    if exclude_genre != "없음":
        clauses.append({"genre": {"$ne": exclude_genre}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
