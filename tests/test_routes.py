import json

import httpx
import pytest
import respx


def _mock_all(siebert_label="POSITIVE", siebert_score=0.95):
    """Register respx mocks for all 7 models in config.yaml order."""
    content = json.dumps({"label": "positive", "score": 0.88})
    respx.post("http://siebert:8000/predict").mock(
        return_value=httpx.Response(200, json={"label": siebert_label, "score": siebert_score})
    )
    respx.post("http://cardiffnlp:8000/predict").mock(
        return_value=httpx.Response(200, json={"label": "positive", "score": 0.80})
    )
    respx.post("http://distilbert:8000/predict").mock(
        return_value=httpx.Response(200, json={"label": "positive", "score": 0.75})
    )
    respx.post("http://vader:8000/predict").mock(
        return_value=httpx.Response(200, json={"label": "positive", "score": 0.91})
    )
    respx.post("http://finbert:8000/predict").mock(
        return_value=httpx.Response(200, json={"label": "positive", "score": 0.82})
    )
    respx.post("http://nlptown:8000/predict").mock(
        return_value=httpx.Response(200, json={"label": "5 stars", "score": 0.71})
    )
    respx.post("http://localhost:8900/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    )


class TestAnalyseEndpoint:
    @respx.mock
    def test_all_models_returned(self, client):
        _mock_all()
        resp = client.post("/analyse", json={"text": "I love it!"})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 7
        models = [r["model"] for r in results]
        assert "siebert/sentiment-roberta-large-english" in models
        assert "cardiffnlp/twitter-roberta-base-sentiment-latest" in models
        assert "lxyuan/distilbert-base-multilingual-cased-sentiments-student" in models
        assert "vader" in models
        assert "ProsusAI/finbert" in models
        assert "nlptown/bert-base-multilingual-uncased-sentiment" in models
        assert "vllm" in models

    @respx.mock
    def test_results_contain_label_score_latency(self, client):
        _mock_all()
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
        respx.post("http://distilbert:8000/predict").mock(
            return_value=httpx.Response(200, json={"label": "positive", "score": 0.75})
        )
        respx.post("http://vader:8000/predict").mock(
            return_value=httpx.Response(200, json={"label": "positive", "score": 0.91})
        )
        respx.post("http://finbert:8000/predict").mock(
            return_value=httpx.Response(200, json={"label": "positive", "score": 0.82})
        )
        respx.post("http://nlptown:8000/predict").mock(
            return_value=httpx.Response(200, json={"label": "5 stars", "score": 0.71})
        )
        content = json.dumps({"label": "positive", "score": 0.88})
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        )
        resp = client.post("/analyse", json={"text": "Great!"})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 7
        siebert = next(r for r in results if r["model"] == "siebert/sentiment-roberta-large-english")
        assert siebert["error"] is not None
        assert siebert["label"] is None

    @respx.mock
    def test_results_in_config_order(self, client):
        _mock_all()
        resp = client.post("/analyse", json={"text": "Nice."})
        models = [r["model"] for r in resp.json()["results"]]
        assert models == [
            "siebert/sentiment-roberta-large-english",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            "vader",
            "ProsusAI/finbert",
            "nlptown/bert-base-multilingual-uncased-sentiment",
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
