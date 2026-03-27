import httpx
import pytest
import respx


class TestSentimentEndpoint:
    @respx.mock
    def test_valid_hf_model(self, client):
        respx.post("http://siebert:8000/predict").mock(
            return_value=httpx.Response(200, json={"label": "POSITIVE", "score": 0.95})
        )
        resp = client.post(
            "/sentiment",
            json={"model": "siebert/sentiment-roberta-large-english", "text": "I love it!"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "positive"
        assert data["score"] == pytest.approx(0.95)
        assert data["model"] == "siebert/sentiment-roberta-large-english"

    @respx.mock
    def test_valid_vllm_model(self, client):
        import json
        content = json.dumps({"label": "negative", "score": 0.88})
        body = {"choices": [{"message": {"content": content}}]}
        respx.post("http://vllm:8900/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        resp = client.post("/sentiment", json={"model": "vllm", "text": "Terrible experience."})
        assert resp.status_code == 200
        assert resp.json()["label"] == "negative"

    def test_unknown_model_returns_400(self, client):
        resp = client.post("/sentiment", json={"model": "nonexistent/model", "text": "hello"})
        assert resp.status_code == 400
        assert "Unknown model" in resp.json()["detail"]

    @respx.mock
    def test_model_server_down_returns_502(self, client):
        respx.post("http://cardiffnlp:8000/predict").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        resp = client.post(
            "/sentiment",
            json={"model": "cardiffnlp/twitter-roberta-base-sentiment-latest", "text": "hello"},
        )
        assert resp.status_code == 502

    def test_missing_fields_returns_422(self, client):
        resp = client.post("/sentiment", json={"model": "vllm"})
        assert resp.status_code == 422

    def test_empty_text_returns_422_not_502(self, client):
        resp = client.post("/sentiment", json={"model": "vllm", "text": ""})
        assert resp.status_code == 422

    def test_whitespace_only_text_returns_422(self, client):
        resp = client.post("/sentiment", json={"model": "vllm", "text": "   "})
        assert resp.status_code == 422
