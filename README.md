# Oracle of Very Useful Predictions

> Peer-reviewed by nobody. Trusted by everyone who has not checked.

A modular Flask platform of rigorously trained machine learning models that predict things
they have no business predicting. Each prediction is its own module with its own dataset,
model, route, and UI.

## Modules

| Module | Predicts | Status |
|---|---|---|
| `hero_villain` | Whether you are a hero or a villain, from your name alone | live |
| `dancability` | How danceable a song is, from its title | planned |
| `username_ban` | Whether a username will get banned | planned |
| `sandwich` | Structural integrity of a sandwich | planned |

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # windows
pip install -r requirements.txt
python -m app.modules.hero_villain.train
python run.py
```

Open http://127.0.0.1:5000

## Tests

```bash
pytest
```

## Limitations

See `docs/PAPER.md`. (It is mostly limitations.)
