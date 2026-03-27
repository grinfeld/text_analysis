import json

import httpx
import pytest
import respx


class TestAnalyseEndpoint:
    @respx.mock
    def test_all_models_returned(self, client):
        content = json.dumps({"label": "positive", "score": 0.88})
        vllm_body = {"choices": [{"message": {"content": content}}]}
        respx.post("http://siebert:8000/predict").mock(
            return_value=httpx.Response(200, json={"label": "POSITIVE", "score": 0.95})
        )
        respx.post("http://cardiffnlp:8000/predict").mock(
            return_value=httpx.Response(200, json={"label": "positive", "score": 0.80})
        )
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=vllm_body)
        )
        resp = client.post("/analyse", json={"text": "I love it!"})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 3
        models = [r["model"] for r in results]
        assert "siebert/sentiment-roberta-large-english" in models
        assert "cardiffnlp/twitter-roberta-base-sentiment-latest" in models
        assert "vllm" in models

    @respx.mock
    def test_results_contain_label_score_latency(self, client):
        respx.post("http://siebert:8000/predict").mock(
            return_value=httpx.Response(200, json={"label": "POSITIVE", "score": 0.95})
        )
        respx.post("http://cardiffnlp:8000/predict").mock(
            return_value=httpx.Response(200, json={"label": "neutral", "score": 0.60})
        )
        content = json.dumps({"label": "negative", "score": 0.72})
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        )
        resp = client.post("/analyse", json={"text": "Some text."})
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            assert "model" in r
            assert "latency_s" in r
            assert r.get("error") is None
            assert r["label"] in ("positive", "neutral", "negative")
            assert 0.0 <= r["score"] <= 1.0

    @respx.mock
    def test_failed_model_does_not_abort_response(self, client):
        respx.post("http://siebert:8000/predict").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        respx.post("http://cardiffnlp:8000/predict").mock(
            return_value=httpx.Response(200, json={"label": "positive", "score": 0.80})
        )
        content = json.dumps({"label": "positive", "score": 0.88})
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        )
        resp = client.post("/analyse", json={"text": "Great!"})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 3
        siebert = next(r for r in results if r["model"] == "siebert/sentiment-roberta-large-english")
        assert siebert["error"] is not None
        assert siebert["label"] is None

    @respx.mock
    def test_results_in_config_order(self, client):
        respx.post("http://siebert:8000/predict").mock(
            return_value=httpx.Response(200, json={"label": "POSITIVE", "score": 0.9})
        )
        respx.post("http://cardiffnlp:8000/predict").mock(
            return_value=httpx.Response(200, json={"label": "positive", "score": 0.8})
        )
        content = json.dumps({"label": "positive", "score": 0.85})
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        )
        resp = client.post("/analyse", json={"text": "Nice."})
        models = [r["model"] for r in resp.json()["results"]]
        assert models == [
            "siebert/sentiment-roberta-large-english",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "vllm",
        ]

    def test_empty_text_returns_422(self, client):
        resp = client.post("/analyse", json={"text": ""})
        assert resp.status_code == 422

    def test_whitespace_text_returns_422(self, client):
        resp = client.post("/analyse", json={"text": "   "})
        assert resp.status_code == 422

    def test_missing_text_returns_422(self, client):
        resp = client.post("/analyse", json={})
        assert resp.status_code == 422
