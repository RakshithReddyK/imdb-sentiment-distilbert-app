def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_empty_text_returns_400(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_predict_whitespace_only_text_returns_400(client):
    resp = client.post("/predict", json={"text": "   "})
    assert resp.status_code == 400


def test_predict_normal_input_returns_label_and_confidence(client):
    resp = client.post(
        "/predict",
        json={"text": "I absolutely loved this movie, it was fantastic."},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["label"] in {"Positive", "Negative"}
    assert 0.0 <= body["confidence"] <= 1.0

    # The stub model always predicts positive with high confidence -- assert
    # the API surfaces that faithfully rather than re-deriving the number.
    assert body["label"] == "Positive"
    assert body["confidence"] > 0.9


def test_predict_missing_field_returns_422(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
