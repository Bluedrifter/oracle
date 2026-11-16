import pytest

from app import create_app
from app.config import Config


@pytest.fixture
def client(tmp_path):
    class TestConfig(Config):
        TESTING = True
        DB_PATH = str(tmp_path / "test.db")

    return create_app(TestConfig).test_client()


def test_home_loads(client):
    assert client.get("/").status_code == 200


def test_prediction_returns_a_verdict(client):
    res = client.post("/hero-villain/", data={"name": "Darth Vader"})
    assert res.status_code == 200
    assert b"confidence" in res.data


def test_empty_name_does_not_crash(client):
    res = client.post("/hero-villain/", data={"name": ""})
    assert res.status_code == 200
    assert b"does not exist" in res.data
