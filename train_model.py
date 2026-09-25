"""
Train a model that predicts the winner of a 1-on-1 Pokemon battle.

Idea: a battle row is just (First_pokemon, Second_pokemon, Winner) by id.
A model can't learn from ids, so for each battle we look up both fighters'
stats and turn the pair into features = the DIFFERENCE of each stat
(first - second). Target = did the first Pokemon win?

Run:  python train_model.py
Output: model/battle_model.pkl  (used by app.py)
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- paths (edit if yours differ) ---
POKEMON_CSV = "data/pokemon.csv"      # stats file (from terminus7/pokemon-challenge)
COMBATS_CSV = "data/combats.csv"      # 50,000 battles: First_pokemon, Second_pokemon, Winner
MODEL_OUT = "model/battle_model.pkl"

# the six base stats we build features from
STAT_COLS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]


def build_features(pokemon: pd.DataFrame, combats: pd.DataFrame):
    """Turn each battle into a row of stat differences + a 0/1 target."""
    # IMPORTANT: combats.csv references fighters by the '#' in the pokemon.csv
    # that ships WITH IT (terminus7/pokemon-challenge), where '#' is a unique
    # 1..800 row id. If '#' is not unique, you're using the wrong pokemon.csv
    # (e.g. one where '#' is the national dex number and repeats for Mega forms)
    # and the ids won't line up. Fail loudly instead of predicting garbage.
    if not pokemon["#"].is_unique:
        raise ValueError(
            "pokemon.csv '#' column is not unique. Use the pokemon.csv bundled "
            "with combats.csv (terminus7/pokemon-challenge), not a national-dex file."
        )

    # index stats by the pokemon id (#) so we can look fighters up fast
    stats = pokemon.set_index("#")[STAT_COLS]

    # pull the stats for each side of every battle, in battle order
    first = stats.loc[combats["First_pokemon"]].reset_index(drop=True)
    second = stats.loc[combats["Second_pokemon"]].reset_index(drop=True)

    # features: first_stat - second_stat, for all six stats
    diff = first.values - second.values
    X = pd.DataFrame(diff, columns=[f"diff_{c}" for c in STAT_COLS])

    # target: 1 if the first pokemon won, else 0
    y = (combats["Winner"] == combats["First_pokemon"]).astype(int)
    return X, y


def main():
    pokemon = pd.read_csv(POKEMON_CSV)
    combats = pd.read_csv(COMBATS_CSV)

    X, y = build_features(pokemon, combats)
    print(f"Battles: {len(X)} | first-wins rate: {y.mean():.3f}")  # ~0.50, balanced

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # depth cap keeps the saved model small (fully-grown trees balloon to 100s of MB)
    model = RandomForestClassifier(
        n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred))

    # save the trained model AND the feature order, so the app builds features
    # exactly the same way at predict time
    joblib.dump({"model": model, "features": list(X.columns), "stat_cols": STAT_COLS},
                MODEL_OUT)
    print(f"Saved -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
