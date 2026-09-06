from dataclasses import replace
import unittest

from ehq.cache import ResponseCache
from ehq.types import InferenceRequest, InferenceResponse, ModelSpec


def _request():
    return InferenceRequest(
        model=ModelSpec(
            name="Model",
            provider="asu",
            provider_model="model-key",
            model_provider="aws",
        ),
        prompt="Question?",
        system_prompt="System",
        temperature=0.0,
        max_tokens=100,
        timeout_seconds=30,
        purpose="answer",
    )


class CacheTests(unittest.TestCase):
    def test_cache_identity_includes_inference_parameters(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            cache = ResponseCache(__import__("pathlib").Path(tmp))
            request = _request()
            self.assertNotEqual(
                cache.key(request, "https://a"),
                cache.key(replace(request, temperature=0.7), "https://a"),
            )
            self.assertNotEqual(
                cache.key(request, "https://a"), cache.key(request, "https://b")
            )
            self.assertNotEqual(
                cache.key(request, "https://a"),
                cache.key(replace(request, purpose="confidence"), "https://a"),
            )
            self.assertNotEqual(
                cache.key(request, "https://a"),
                cache.key(replace(request, timeout_seconds=31), "https://a"),
            )
            thinking_disabled = replace(
                request,
                model=replace(request.model, thinking_mode="disabled"),
            )
            self.assertNotEqual(
                cache.key(request, "https://a"),
                cache.key(thinking_disabled, "https://a"),
            )

    def test_successful_response_round_trip(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            cache = ResponseCache(__import__("pathlib").Path(tmp))
            request = _request()
            response = InferenceResponse(
                ok=True,
                text="answer",
                requested_model="model-key",
                resolved_model="model-key",
                provider="asu",
            )
            cache.put(request, "https://a", response)
            loaded = cache.get(request, "https://a")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.text, "answer")
            self.assertTrue(loaded.cache_hit)

    def test_failures_are_not_cacheable(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            cache = ResponseCache(__import__("pathlib").Path(tmp))
            response = InferenceResponse(
                ok=False,
                text=None,
                requested_model="model-key",
                resolved_model=None,
                provider="asu",
                error_type="timeout",
            )
            with self.assertRaises(ValueError):
                cache.put(_request(), "https://a", response)
