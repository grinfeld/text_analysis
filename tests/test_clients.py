import json

import httpx
import pytest
import respx

from text_analysis.clients.base import ModelClientError
from text_analysis.clients.model_server import ModelServerClient

_SIEBERT_LABELS: dict[str, str] = {"POSITIVE": "positive", "NEGATIVE": "negative"}
from text_analysis.clients.vllm import (
    VllmClient,
    TOPIC_SYSTEM_PROMPT,
    TOPIC_USER_PROMPT_TEMPLATE,
)


@pytest.fixture
def hf_http():
    return httpx.AsyncClient()


@pytest.fixture
def siebert_client(hf_http):
    return ModelServerClient(
        model_name="siebert/sentiment-roberta-large-english",
        base_url="http://siebert:8000",
        http_client=hf_http,
        label_map=_SIEBERT_LABELS,
    )


@pytest.fixture
def vllm_http():
    return httpx.AsyncClient()


@pytest.fixture
def vllm_client(vllm_http):
    return VllmClient(
        model_name="vllm",
        base_url="http://localhost:8900",
        model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        http_client=vllm_http,
    )


@pytest.fixture
def vllm_topic_client(vllm_http):
    return VllmClient(
        model_name="vllm-topic",
        base_url="http://localhost:8900",
        model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        http_client=vllm_http,
        task="topic",
        system_prompt=TOPIC_SYSTEM_PROMPT,
        user_prompt_template=TOPIC_USER_PROMPT_TEMPLATE,
        valid_labels=frozenset(),  # open-ended — accept any slug
    )


def _ms_response(labels: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"labels": labels})


class TestModelServerClient:
    @respx.mock
    async def test_predict_positive(self, siebert_client):
        respx.post("http://siebert:8000/predict").mock(
            return_value=_ms_response([{"label": "POSITIVE", "score": 0.98}])
        )
        result = await siebert_client.predict("I love this!")
        assert result.label == "positive"
        assert result.score == pytest.approx(0.98)

    @respx.mock
    async def test_predict_negative(self, siebert_client):
        respx.post("http://siebert:8000/predict").mock(
            return_value=_ms_response([{"label": "NEGATIVE", "score": 0.92}])
        )
        result = await siebert_client.predict("This is terrible.")
        assert result.label == "negative"

    @respx.mock
    async def test_http_error_raises_client_error(self, siebert_client):
        respx.post("http://siebert:8000/predict").mock(
            return_value=httpx.Response(503, text="service unavailable")
        )
        with pytest.raises(ModelClientError):
            await siebert_client.predict("hello")

    @respx.mock
    async def test_malformed_response_raises_client_error(self, siebert_client):
        respx.post("http://siebert:8000/predict").mock(
            return_value=httpx.Response(200, json={"unexpected": "shape"})
        )
        with pytest.raises(ModelClientError):
            await siebert_client.predict("hello")

    @respx.mock
    async def test_timeout_raises_client_error(self, siebert_client):
        respx.post("http://siebert:8000/predict").mock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(ModelClientError):
            await siebert_client.predict("hello")

    @respx.mock
    async def test_unknown_label_falls_back_to_neutral(self, siebert_client):
        respx.post("http://siebert:8000/predict").mock(
            return_value=_ms_response([{"label": "UNKNOWN_LABEL", "score": 0.5}])
        )
        result = await siebert_client.predict("hmm")
        assert result.label == "neutral"

    @respx.mock
    async def test_empty_labels_list_raises_client_error(self, siebert_client):
        respx.post("http://siebert:8000/predict").mock(
            return_value=httpx.Response(200, json={"labels": []})
        )
        with pytest.raises(ModelClientError):
            await siebert_client.predict("hello")

    @respx.mock
    async def test_empty_label_map_passes_label_through(self, hf_http):
        # Zero-shot models have no label_map — raw labels are returned as-is
        client = ModelServerClient(
            model_name="deberta",
            base_url="http://deberta:8000",
            http_client=hf_http,
            label_map={},
            task="topic",
        )
        respx.post("http://deberta:8000/predict").mock(
            return_value=_ms_response([
                {"label": "financial_crime", "score": 0.91},
                {"label": "money_laundering", "score": 0.65},
                {"label": "sanctions_evasion", "score": 0.42},
            ])
        )
        result = await client.predict("Shell company transferred funds offshore.")
        assert result.label == "financial_crime"
        assert len(result.labels) == 3
        assert result.labels[1] == ("money_laundering", pytest.approx(0.65), None)

    @respx.mock
    async def test_source_relation_target_concatenated_with_text(self, siebert_client):
        captured = {}

        async def capture(request):
            captured["body"] = request.read()
            return _ms_response([{"label": "POSITIVE", "score": 0.9}])

        respx.post("http://siebert:8000/predict").mock(side_effect=capture)
        await siebert_client.predict("some text", source="Russia", relation="attacked", target="Ukraine")
        import json as _json
        sent = _json.loads(captured["body"])
        assert sent["text"] == "Russia attacked Ukraine some text"

    @respx.mock
    async def test_partial_entity_fields_concatenated(self, siebert_client):
        captured = {}

        async def capture(request):
            captured["body"] = request.read()
            return _ms_response([{"label": "POSITIVE", "score": 0.9}])

        respx.post("http://siebert:8000/predict").mock(side_effect=capture)
        await siebert_client.predict("details here", source="APT28")
        import json as _json
        sent = _json.loads(captured["body"])
        assert sent["text"] == "APT28 details here"


class TestVllmClient:
    def _make_vllm_response(self, label: str, score: float) -> httpx.Response:
        content = json.dumps({"label": label, "score": score})
        body = {"choices": [{"message": {"content": content}}]}
        return httpx.Response(200, json=body)

    def _make_vllm_topic_response(self, labels: list[dict]) -> httpx.Response:
        content = json.dumps({"labels": labels})
        body = {"choices": [{"message": {"content": content}}]}
        return httpx.Response(200, json=body)

    @respx.mock
    async def test_predict_positive(self, vllm_client):
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=self._make_vllm_response("positive", 0.9)
        )
        result = await vllm_client.predict("Great product!")
        assert result.label == "positive"
        assert result.score == pytest.approx(0.9)

    @respx.mock
    async def test_predict_neutral(self, vllm_client):
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=self._make_vllm_response("neutral", 0.7)
        )
        result = await vllm_client.predict("The sky is blue.")
        assert result.label == "neutral"

    @respx.mock
    async def test_unknown_label_falls_back_to_neutral(self, vllm_client):
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=self._make_vllm_response("mixed", 0.5)
        )
        result = await vllm_client.predict("It's complicated.")
        assert result.label == "neutral"

    @respx.mock
    async def test_bad_json_raises_client_error(self, vllm_client):
        body = {"choices": [{"message": {"content": "not json at all"}}]}
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        with pytest.raises(ModelClientError):
            await vllm_client.predict("hello")

    @respx.mock
    async def test_http_error_raises_client_error(self, vllm_client):
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="internal error")
        )
        with pytest.raises(ModelClientError):
            await vllm_client.predict("hello")

    @respx.mock
    async def test_score_clamped_to_unit_interval(self, vllm_client):
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=self._make_vllm_response("positive", 1.5)
        )
        result = await vllm_client.predict("amazing!")
        assert result.score == pytest.approx(1.0)

    @respx.mock
    async def test_topic_returns_ranked_list(self, vllm_topic_client):
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=self._make_vllm_topic_response([
                {"label": "financial_crime", "score": 0.88},
                {"label": "money_laundering", "score": 0.65},
                {"label": "sanctions_evasion", "score": 0.42},
            ])
        )
        result = await vllm_topic_client.predict("Shell company transferred funds offshore.")
        assert result.label == "financial_crime"
        assert len(result.labels) == 3
        assert result.labels[1] == ("money_laundering", pytest.approx(0.65), None)

    @respx.mock
    async def test_missing_score_raises_client_error(self, vllm_client):
        content = json.dumps({"label": "positive"})  # no "score"
        body = {"choices": [{"message": {"content": content}}]}
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=body)
        )
        with pytest.raises(ModelClientError):
            await vllm_client.predict("hello")

    @respx.mock
    async def test_topic_empty_labels_list_raises_client_error(self, vllm_topic_client):
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=self._make_vllm_topic_response([])
        )
        with pytest.raises(ModelClientError):
            await vllm_topic_client.predict("hello")

    @respx.mock
    async def test_topic_novel_slug_passes_through(self, vllm_topic_client):
        # Open-ended topic mode accepts any slug the model proposes
        respx.post("http://localhost:8900/v1/chat/completions").mock(
            return_value=self._make_vllm_topic_response([
                {"label": "border_security", "score": 0.82},
                {"label": "covert_operation", "score": 0.50},
                {"label": "recruitment", "score": 0.30},
            ])
        )
        result = await vllm_topic_client.predict("Customs officials intercepted a vehicle at the border.")
        assert result.label == "border_security"
