import unittest

from ehq.clients.openai import OpenAIResponsesClient
from ehq.clients.openai_compatible import OpenAICompatibleClient
from ehq.types import InferenceRequest, ModelSpec


class FakeHTTPResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = "fake response"

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return self.response


def request(model):
    return InferenceRequest(
        model=model,
        prompt="Question?",
        system_prompt="You are helpful.",
        temperature=0,
        max_tokens=100,
        timeout_seconds=30,
    )


class OpenAIResponsesClientTests(unittest.TestCase):
    def test_uses_responses_api_and_extracts_output_text(self):
        session = FakeSession(
            FakeHTTPResponse(
                {
                    "id": "resp_1",
                    "model": "gpt-4.1-mini",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "Answer"}
                            ],
                        }
                    ],
                    "usage": {"output_tokens": 3},
                }
            )
        )
        model = ModelSpec(
            name="openai",
            provider="openai",
            provider_model="gpt-4.1-mini",
            reasoning_mode="disabled",
        )
        client = OpenAIResponsesClient(
            api_key="secret",
            session=session,
            max_retries=1,
            request_delay_seconds=0,
        )
        response = client.query(request(model), use_cache=False)
        self.assertTrue(response.ok)
        self.assertEqual(response.text, "Answer")
        endpoint, call = session.calls[0]
        self.assertEqual(endpoint, "https://api.openai.com/v1/responses")
        self.assertFalse(call["json"]["store"])
        self.assertEqual(call["json"]["temperature"], 0)
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret")

    def test_strict_resolved_model_mismatch_is_terminal(self):
        session = FakeSession(
            FakeHTTPResponse(
                {
                    "model": "different-model",
                    "output_text": "Answer",
                    "usage": {},
                }
            )
        )
        model = ModelSpec(
            name="strict",
            provider="openai",
            provider_model="requested-model",
            model_identity_policy="strict",
            reasoning_mode="disabled",
        )
        client = OpenAIResponsesClient(
            api_key="secret",
            session=session,
            max_retries=2,
            request_delay_seconds=0,
        )
        response = client.query(request(model), use_cache=False)
        self.assertFalse(response.ok)
        self.assertEqual(response.error_type, "model_mismatch")
        self.assertEqual(response.attempts, 1)


class OpenAICompatibleClientTests(unittest.TestCase):
    def test_huggingface_shape_and_usage_are_preserved(self):
        session = FakeSession(
            FakeHTTPResponse(
                {
                    "id": "chat_1",
                    "model": "org/model:fastest",
                    "choices": [{"message": {"content": "Answer"}}],
                    "usage": {"completion_tokens": 4},
                }
            )
        )
        model = ModelSpec(
            name="hf",
            provider="huggingface",
            provider_model="org/model:fastest",
            reasoning_mode="disabled",
        )
        client = OpenAICompatibleClient(
            provider_name="huggingface",
            base_url="https://router.huggingface.co/v1",
            api_key_env="HF_TOKEN",
            requires_api_key=True,
            api_key="hf_secret",
            session=session,
            max_retries=1,
            request_delay_seconds=0,
        )
        response = client.query(request(model), use_cache=False)
        self.assertTrue(response.ok)
        self.assertEqual(response.usage["completion_tokens"], 4)
        endpoint, call = session.calls[0]
        self.assertEqual(
            endpoint, "https://router.huggingface.co/v1/chat/completions"
        )
        self.assertEqual(call["json"]["messages"][1]["content"], "Question?")

    def test_local_route_does_not_require_authorization_header(self):
        session = FakeSession(
            FakeHTTPResponse(
                {
                    "model": "local-model",
                    "choices": [{"message": {"content": "Answer"}}],
                }
            )
        )
        model = ModelSpec(
            name="local",
            provider="openai_compatible",
            provider_model="local-model",
            base_url="http://localhost:8000/v1",
            requires_api_key=False,
            reasoning_mode="disabled",
        )
        client = OpenAICompatibleClient(
            provider_name="openai_compatible",
            base_url=model.base_url,
            api_key_env=None,
            requires_api_key=False,
            session=session,
            max_retries=1,
            request_delay_seconds=0,
        )
        response = client.query(request(model), use_cache=False)
        self.assertTrue(response.ok)
        self.assertNotIn("Authorization", session.calls[0][1]["headers"])


if __name__ == "__main__":
    unittest.main()
