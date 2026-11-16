"""Every prediction ever made gets logged. The oracle remembers."""
import sqlite3
from pathlib import Path


def _connect(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: str) -> None:
    with _connect(db_path) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module TEXT NOT NULL,
                input TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )


def log_prediction(db_path, module, text, verdict, confidence) -> None:
    with _connect(db_path) as con:
        con.execute(
            "INSERT INTO predictions (module, input, verdict, confidence) VALUES (?, ?, ?, ?)",
            (module, text, verdict, confidence),
        )


def recent(db_path, module=None, limit=10):
    query = "SELECT input, verdict, confidence FROM predictions"
    args = []
    if module:
        query += " WHERE module = ?"
        args.append(module)
    query += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _connect(db_path) as con:
        return con.execute(query, args).fetchall()
