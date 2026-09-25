"""
Flask app for the Pokemon Battle Predictor.

Pick two Pokemon in the browser, hit Battle, and the trained model predicts
who wins. Run train_model.py first to create model/battle_model.pkl.

Run:  python app.py   ->  open http://127.0.0.1:5000
"""

import os
import re
import glob
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

POKEMON_CSV = "data/pokemon.csv"
MODEL_PATH = "model/battle_model.pkl"
IMAGES_DIR = "static/images"

app = Flask(__name__)

# --- load model + pokemon data once at startup ---
bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
FEATURES = bundle["features"]      # ['diff_HP', 'diff_Attack', ...]
STAT_COLS = bundle["stat_cols"]    # ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']

pokemon = pd.read_csv(POKEMON_CSV)

# --- sprite resolution ---
# sprite files (from the image dataset) are lowercase, hyphenated pokeapi-style
# names; the stats file spells names differently, so try a couple of candidates
# and fall back to a placeholder (Mega forms have no sprite in the dataset).
_have = {os.path.basename(p) for p in glob.glob(os.path.join(IMAGES_DIR, "*.png"))}


def resolve_sprite(name):
    if pd.isna(name):
        return None
    n = str(name).lower().strip().replace("\u2640", "-f").replace("\u2642", "-m")
    api = re.sub(r"-+", "-", re.sub(r"\s+", "-", re.sub(r"[.'\u2019]", "", n)))
    aggressive = re.sub(r"[^a-z0-9]", "", n)
    for cand in (f"{api}.png", f"{aggressive}.png"):
        if cand in _have:
            return f"/static/images/{cand}"
    return None  # UI shows a placeholder


# build the list the UI needs: id, name, stats, sprite url
POKEDEX = []
for _, row in pokemon.iterrows():
    name = row["Name"] if pd.notna(row["Name"]) else f"Unknown #{int(row['#'])}"
    POKEDEX.append({
        "id": int(row["#"]),
        "name": name,
        "stats": {c: int(row[c]) for c in STAT_COLS},
        "img": resolve_sprite(row["Name"]),   # None -> placeholder in the UI
    })

BY_ID = {p["id"]: p for p in POKEDEX}


@app.route("/")
def index():
    return render_template("index.html", pokedex=POKEDEX)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    first, second = BY_ID.get(data["first"]), BY_ID.get(data["second"])
    if not first or not second:
        return jsonify({"error": "unknown pokemon"}), 400

    # build features exactly like training: first_stat - second_stat, in order
    diffs = [[first["stats"][c] - second["stats"][c] for c in STAT_COLS]]
    row = pd.DataFrame(diffs, columns=FEATURES)

    prob_first = float(MODEL.predict_proba(row)[0][1])  # P(first wins)
    winner = first if prob_first >= 0.5 else second
    win_prob = prob_first if prob_first >= 0.5 else 1 - prob_first

    return jsonify({
        "winner_id": winner["id"],
        "winner_name": winner["name"],
        "win_prob": round(win_prob * 100, 1),
    })


if __name__ == "__main__":
    app.run(debug=True)
