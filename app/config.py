from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    SECRET_KEY = "the-oracle-sees-all"
    DB_PATH = str(BASE_DIR / "instance" / "oracle.db")
