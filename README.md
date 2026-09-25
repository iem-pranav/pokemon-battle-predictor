# Pokémon Battle Predictor

Predicts the winner of a 1-on-1 Pokémon battle from the two fighters' base stats,
served through a Pokédex-style web app: pick two Pokémon, hit **Battle**, and the
model tells you who it bets on and how confident it is.

Unlike guessing whether a Pokémon is "legendary" (a fixed label), a battle outcome
is genuinely unknown for two arbitrary Pokémon — so this is a real prediction task.

## How it works

A battle is recorded as `(First_pokemon, Second_pokemon, Winner)` by id — no stats
attached. A model can't learn from ids, so for each battle we:

1. look up both fighters' stats,
2. build features as the **difference** of each stat (`first − second`) — this
   captures who is stronger where,
3. set the target to `1` if the first Pokémon won, else `0`.

The classes are roughly balanced (~50/50), so accuracy is a meaningful score here.
A `RandomForestClassifier` scores **~95% accuracy** on the 50,000 real battles
(precision and recall both ~0.95).

## Project structure

```
pokemon-battle-predictor/
├── train_model.py        # build features, train, save model/battle_model.pkl
├── app.py                # Flask app: serves the UI + /predict endpoint
├── templates/index.html  # the Pokédex battle screen
├── static/style.css      # styling
├── static/images/        # Pokémon sprites (downloaded, gitignored)
├── data/                 # CSVs (downloaded, gitignored)
├── model/                # trained model (generated, gitignored)
├── requirements.txt
└── README.md
```

## Setup

> If you're using the packaged zip, `data/` (stats + combats) and
> `static/images/` (sprites) are already filled in — skip to step 3 and just
> train + run. Steps 1–2 are for anyone cloning the repo from GitHub, where those
> files are gitignored. Sprites cover ~88% of Pokémon; Mega forms have no sprite
> in the dataset and show a placeholder.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**1. Battle data** — [terminus7/pokemon-challenge](https://www.kaggle.com/datasets/terminus7/pokemon-challenge).
Put `pokemon.csv` and `combats.csv` into `data/`.
> ⚠️ Use the `pokemon.csv` that ships **with** `combats.csv`. Its `#` column is a
> unique 1–800 id that the battle ids point to. A different Pokémon file (where `#`
> is the national-dex number and repeats for Mega forms) will not line up — the
> training script will stop and tell you if that happens.

**2. Sprites** — [vishalsubbiah/pokemon-images-and-types](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types).
Put the image files into `static/images/` (named like `bulbasaur.png`). Missing
sprites (some Mega/alternate forms) fall back to a placeholder — the app still works.

**3. Train, then run**

```bash
python train_model.py     # creates model/battle_model.pkl
python app.py             # open http://127.0.0.1:5000
```

## Limitations & next steps

- Uses only base stats — it doesn't know type effectiveness (Water beats Fire), so
  it can be wrong on strong type matchups even when stats say otherwise.
- **v2 ideas:** add type-matchup features; try the decision threshold / calibration;
  show each fighter's predicted win probability across all opponents.

## Data credits

Battle + stats data: *terminus7 / Pokémon- Weedle's Cave*.
Sprites: *vishalsubbiah / Pokémon Images and Types*.
