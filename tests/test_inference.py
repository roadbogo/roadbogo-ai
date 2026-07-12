from fastapi.testclient import TestClient

from app.main import app


def test_inference() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/inference/",
        json={"route_id": "sample-route", "features": {"speed": 0.5, "density": 0.9}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "route_id": "sample-route",
        "score": 0.7,
        "label": "attention",
    }
