from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = {
    "genres": "genre",
    "categories": "category",
    "steamspy_tags": "tag",
}


def normalize_labels(value, prefix):
    tokens = []
    for label in value.split(";"):
        label = label.strip().lower()
        label = re.sub(r"[\W_]+", "_", label).strip("_")
        if label:
            token = f"{prefix}_{label}"
            if token not in tokens:
                tokens.append(token)
    return " ".join(tokens)


def prepare_data():
    data = pd.read_csv(ROOT / "data" / "steam.csv")
    rows_before = len(data)

    if data["appid"].isna().any() or data["appid"].duplicated().any():
        raise ValueError("appid must be present and unique for every game.")

    processed = pd.DataFrame(index=data.index)
    for column, prefix in FEATURE_COLUMNS.items():
        data[column] = data[column].fillna("")
        processed[column] = data[column].map(
            lambda value: normalize_labels(value, prefix)
        )

    data["content_features"] = processed.agg(" ".join, axis=1).str.split().str.join(" ")
    missing_name = data["name"].fillna("").str.strip().eq("")
    empty_content = data["content_features"].eq("")
    cleaned = data.loc[~(missing_name | empty_content)].copy()

    output = ROOT / "data" / "steam_clean.csv"
    cleaned.to_csv(output, index=False)

    print(f"Rows before: {rows_before:,}")
    print(f"Rows after: {len(cleaned):,}")
    print(f"Rows removed: {rows_before - len(cleaned):,}")
    print(f"Missing name: {missing_name.sum():,}")
    print(f"Empty content features: {empty_content.sum():,}")
    print(f"Both reasons: {(missing_name & empty_content).sum():,}")
    print(f"Games sharing a name: {cleaned['name'].duplicated(keep=False).sum():,}")
    print(f"Repeated name groups: {cleaned.loc[cleaned['name'].duplicated(keep=False), 'name'].nunique():,}")
    print(f"Additional rows with repeated names: {cleaned['name'].duplicated().sum():,}")
    print("Example content features:")
    for _, game in cleaned.head(5).iterrows():
        print(f"{game['name']}: {game['content_features']}")


if __name__ == "__main__":
    prepare_data()
