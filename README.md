# Game Recommendation System

## Project Overview

A terminal-based game recommender that uses Steam content metadata to find
similar games rather than relying on user ratings.

## Objective

Recommend up to five similar games based on genres, categories, and SteamSpy tags.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Git/GitHub

## Dataset

The Steam Store Games dataset (`data/steam.csv`) contains 27,075 entries, including
some non-game applications, with release dates from 1997 to May 2019. Similarity
uses only genres, categories, and SteamSpy tags; ratings, owners, price, and
playtime are kept as metadata.

## Data Preparation

`scripts/prepare_data.py` reads the original dataset and writes
`data/steam_clean.csv` without changing `steam.csv`.

- Fill missing content features with empty strings and remove rows only if the
  name is missing or all content features are empty after normalization.
- Split labels on `;`, lowercase them, trim spaces, and replace spaces and
  punctuation with underscores, such as `open_world`.
- Add feature prefixes such as `genre_action`, `category_single_player`, and
  `tag_open_world`, then combine tokens into `content_features`.
- Keep `appid` as the unique identifier, retain duplicate names, and preserve
  other metadata. All 27,075 rows were retained.

## Recommendation Method

TF-IDF turns `content_features` into numeric vectors, giving more weight to less
common labels. Cosine similarity compares these vectors to find games with
similar content.

```text
Game selected → content features → TF-IDF vectors → cosine similarity → Top 5 recommendations
```

The selected game is excluded from results. Game-name search is normalized, and
duplicate titles can be distinguished using `appid`.

Results include the game name, similarity score, genres, developer, and publisher.
Similarity scores are also grouped into simple labels to make the results easier to interpret.
Lower-scoring results are retained.

## Example

Searching for `Dark Souls III` matches **DARK SOULS™ III**:

| Recommendation | Similarity score |
|---|---:|
| DARK SOULS™ II | 0.9648 |
| DARK SOULS™ II: Scholar of the First Sin | 0.9648 |
| DARK SOULS™: REMASTERED | 0.9039 |
| INFERNO CLIMBER | 0.7389 |
| Tormentum - Dark Sorrow | 0.5908 |

These scores describe feature similarity, not game quality or user preference.

## Evaluation

Genre Match Rate checks whether recommendations share at least one genre with
the selected game:

```text
Genre Match Rate = recommendations sharing at least one genre / total recommendations
```

Five games were evaluated using their top five recommendations:

| Selected game | Genre matches | Genre Match Rate |
|---|---:|---:|
| DARK SOULS™ III | 4/5 | 80% |
| Counter-Strike | 5/5 | 100% |
| Stardew Valley | 5/5 | 100% |
| Grand Theft Auto V | 4/5 | 80% |
| Sid Meier's Civilization® V | 5/5 | 100% |

**Average Genre Match Rate: 92%.** This is a sanity check, not recommendation
accuracy measured by user ratings. High overlap is expected because genres are
also input features.

## Limitations

- The system does not use individual user preferences or ratings.
- Similarity depends on limited Steam metadata; shared labels can still lead to
  recommendations with quite different gameplay.
- The dataset includes non-game applications and does not cover newer releases.
- Evaluation covers only five selected games, and genre overlap does not measure
  user satisfaction.

## How to Run

Run these commands from the project root.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Prepare the dataset:

```bash
python scripts/prepare_data.py
```

Start the interactive game search:

```bash
python scripts/recommend_games.py
```

Or provide a name directly, with an optional `appid` to distinguish matches:

```bash
python scripts/recommend_games.py "Dark Souls III"
python scripts/recommend_games.py "Dark Souls III" --appid 374320
```

Run the evaluation:

```bash
python scripts/evaluate_recommender.py
```

## Project Structure

```text
game-recommendation-system/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── steam.csv
│   └── steam_clean.csv
└── scripts/
    ├── prepare_data.py
    ├── recommend_games.py
    └── evaluate_recommender.py
```
