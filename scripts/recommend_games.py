import argparse
from difflib import get_close_matches
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "steam_clean.csv"


def find_game(games, name, appid=None):
    normalized_names = games["name"].str.strip().str.casefold()
    query = name.strip().casefold()
    matches = games.loc[normalized_names.eq(query)]

    if matches.empty:
        names = dict(zip(normalized_names, games["name"]))
        suggestions = get_close_matches(query, names, n=5, cutoff=0.4)
        print(f'No exact match for "{name}".')
        if suggestions:
            print("Did you mean:")
            for suggestion in suggestions:
                print(f"  {names[suggestion]}")
        else:
            print("Try another game name.")
        return None

    if appid is not None:
        matches = matches.loc[matches["appid"].eq(appid)]
        if matches.empty:
            print(f'No game named "{name}" has appid {appid}.')
            return None

    if len(matches) > 1:
        print("Multiple games match. Select one by appid:")
        for game in matches.itertuples():
            print(f"  {game.appid}: {game.name} | Developer: {game.developer or 'Unknown'}")
        try:
            selected_appid = int(input("appid: ").strip())
        except ValueError:
            print("Please enter a numeric appid.")
            return None
        matches = matches.loc[matches["appid"].eq(selected_appid)]
        if matches.empty:
            print("That appid is not one of the matching games.")
            return None

    return matches.index[0]


def recommend_games(games, vectors, selected_index):
    scores = cosine_similarity(vectors[selected_index], vectors).ravel()
    scored = games.assign(similarity_score=scores).drop(index=selected_index)
    return scored.sort_values(
        ["similarity_score", "appid"], ascending=[False, True]
    ).head(5)


def main():
    parser = argparse.ArgumentParser(description="Find five similar Steam games.")
    parser.add_argument("name", nargs="?", help="Exact game name (case-insensitive)")
    parser.add_argument("--appid", type=int, help="Select an appid when names match multiple games")
    args = parser.parse_args()

    games = pd.read_csv(DATA_PATH).fillna("").reset_index(drop=True)
    name = args.name if args.name is not None else input("Game name: ")
    selected_index = find_game(games, name, args.appid)
    if selected_index is None:
        return

    vectors = TfidfVectorizer().fit_transform(games["content_features"])
    recommendations = recommend_games(games, vectors, selected_index)
    selected = games.iloc[selected_index]
    print(f"\nGames similar to {selected['name']} (appid {selected['appid']}):")
    for rank, game in enumerate(recommendations.itertuples(), start=1):
        print(f"\n{rank}. {game.name} (appid {game.appid})")
        print(f"   Similarity: {game.similarity_score:.4f}")
        print(f"   Genres: {game.genres or 'Unknown'}")
        print(f"   Developer: {game.developer or 'Unknown'}")
        print(f"   Publisher: {game.publisher or 'Unknown'}")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nSearch cancelled.")
