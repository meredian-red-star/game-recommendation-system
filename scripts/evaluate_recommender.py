from statistics import mean

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from recommend_games import DATA_PATH, find_game, recommend_games


TEST_GAMES = [
    ("Dark Souls III", 374320),
    ("Counter-Strike", 10),
    ("Stardew Valley", 413150),
    ("Grand Theft Auto V", 271590),
    ("Sid Meier's Civilization V", 8930),
]


def genre_set(value):
    return {genre.strip().casefold() for genre in value.split(";") if genre.strip()}


def genre_match_rate(selected_genres, recommendations):
    genres = genre_set(selected_genres)
    matches = recommendations["genres"].map(
        lambda value: bool(genres & genre_set(value))
    )
    rate = matches.sum() / len(matches) if len(matches) else 0.0
    return matches, rate


def main():
    games = pd.read_csv(DATA_PATH).fillna("").reset_index(drop=True)
    vectors = TfidfVectorizer().fit_transform(games["content_features"])
    rates = []

    print("Genre Match Rate = recommendations sharing at least one genre / total recommendations")
    print("This is a basic sanity check, not a complete measure of recommendation quality.")
    print("Genres are also input features, so high overlap is expected. This is not user-rated accuracy.")

    for name, appid in TEST_GAMES:
        selected_index = find_game(games, name, appid)
        if selected_index is None:
            print(f"Skipping unavailable game: {name} (appid {appid})")
            continue

        selected = games.iloc[selected_index]
        recommendations = recommend_games(games, vectors, selected_index)
        matches, rate = genre_match_rate(selected["genres"], recommendations)
        rates.append(rate)

        print(f"\n{selected['name']} (appid {selected['appid']})")
        print(f"Genres: {selected['genres']}")
        for rank, (index, game) in enumerate(recommendations.iterrows(), start=1):
            shared = "Yes" if matches.loc[index] else "No"
            print(f"  {rank}. {game['name']} (appid {game['appid']})")
            print(f"     Similarity: {game['similarity_score']:.4f} | Shared genre: {shared}")
            print(f"     Genres: {game['genres']}")
        print(f"Genre Match Rate: {matches.sum()}/{len(recommendations)} = {rate:.0%}")

    if not rates:
        raise SystemExit("No test games were evaluated.")
    print(f"\nGames evaluated: {len(rates)}/{len(TEST_GAMES)}")
    print(f"Average Genre Match Rate: {mean(rates):.0%}")


if __name__ == "__main__":
    main()
