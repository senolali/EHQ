import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ehq.artifacts import write_artifact_catalog


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VERIFY = _load("verify_route_from_run")


def _record(*, valid12=True, valid3=True, resolved="gpt5_4_mini", thinking=0):
    def envelope(ok):
        return {
            "ok": ok,
            "text": "x" if ok else None,
            "requested_model": "gpt5_4_mini",
            "resolved_model": resolved if ok else None,
            "usage": {"output_token_details": {"thinking": thinking}},
            "metadata": {"ehq_observed_reasoning_tokens": thinking},
        }

    return {
        "model": "GPT-5.4-mini",
        "question_id": "Q",
        "valid_for_ehq12": valid12,
        "valid_for_ehq3": valid3,
        "answer_response": envelope(valid12),
        "confidence_response": envelope(valid3),
        "classification": {"label": "CONFIDENT_WRONG", "is_correct": False},
    }


class VerifyRouteTests(unittest.TestCase):
    def _assess(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "records.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
            )
            (run / "manifest.json").write_text(
                json.dumps(
                    {
                        "created_at_utc": "2026-08-05T20:00:00+00:00",
                        "run": {"fingerprint": "abc"},
                    }
                ),
                encoding="utf-8",
            )
            write_artifact_catalog(run)
            options = {"min_valid": 0.99, "min_confidence": 0.99}
            options.update(kwargs)
            return VERIFY.assess(run, "GPT-5.4-mini", **options)

    def test_a_clean_run_verifies(self):
        report = self._assess([_record() for _ in range(100)])
        self.assertTrue(report["verified"], report["failures"])
        self.assertEqual(report["valid_rate"], 1.0)
        self.assertEqual(report["confidence_rate"], 1.0)
        self.assertEqual(report["max_reasoning_tokens"], 0)

    def test_missing_confidence_below_threshold_refuses(self):
        rows = [_record() for _ in range(95)] + [
            _record(valid3=False) for _ in range(5)
        ]
        report = self._assess(rows)
        self.assertFalse(report["verified"])
        self.assertTrue(any("confidence" in f for f in report["failures"]))

    def test_answer_failures_below_threshold_refuse(self):
        rows = [_record() for _ in range(95)] + [
            _record(valid12=False, valid3=False) for _ in range(5)
        ]
        report = self._assess(rows)
        self.assertFalse(report["verified"])
        self.assertTrue(any("EHQ1/EHQ2" in f for f in report["failures"]))

    def test_a_route_resolving_elsewhere_refuses(self):
        rows = [_record() for _ in range(99)] + [_record(resolved="gpt5_mini")]
        report = self._assess(rows)
        self.assertFalse(report["verified"])
        self.assertTrue(any("route identity" in f for f in report["failures"]))

    def test_any_reported_reasoning_token_refuses(self):
        rows = [_record() for _ in range(99)] + [_record(thinking=12)]
        report = self._assess(rows)
        self.assertFalse(report["verified"])
        self.assertTrue(any("reasoning" in f for f in report["failures"]))
        self.assertEqual(report["max_reasoning_tokens"], 12)

    def test_a_run_without_a_catalog_is_refused_not_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "records.jsonl").write_text(
                "\n".join(json.dumps(_record()) for _ in range(100)),
                encoding="utf-8",
            )
            (run / "manifest.json").write_text(
                json.dumps({"created_at_utc": "x", "run": {}}), encoding="utf-8"
            )
            report = VERIFY.assess(
                run, "GPT-5.4-mini", min_valid=0.99, min_confidence=0.99
            )
        self.assertFalse(report["verified"])
        self.assertTrue(any("catalog" in f for f in report["failures"]))

    def test_thresholds_are_reported_with_the_decision(self):
        report = self._assess([_record() for _ in range(10)], min_valid=0.5)
        self.assertEqual(report["thresholds"]["min_valid"], 0.5)
        self.assertEqual(report["thresholds"]["min_confidence"], 0.99)


class RegistryWriteTests(unittest.TestCase):
    REPORT = {
        "model": "GPT-5.4-mini",
        "run_created_at_utc": "2026-08-05T20:00:00+00:00",
        "resolved_routes": {"gpt5_4_mini": 200},
        "max_reasoning_tokens": 0,
    }

    def test_verification_fields_are_written_from_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text(
                json.dumps(
                    {
                        "models": [
                            {"name": "Other", "operational_status": "verified"},
                            {
                                "name": "GPT-5.4-mini",
                                "operational_status": "unverified",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            VERIFY.apply_to_registry(path, self.REPORT)
            entry = [
                m
                for m in json.loads(path.read_text(encoding="utf-8"))["models"]
                if m["name"] == "GPT-5.4-mini"
            ][0]
            self.assertEqual(entry["operational_status"], "verified")
            self.assertEqual(
                entry["endpoint_verified_resolved_model"], "gpt5_4_mini"
            )
            self.assertEqual(entry["model_identity_policy"], "strict")
            self.assertEqual(entry["non_reasoning_observed_thinking_tokens"], 0)
            self.assertEqual(
                entry["endpoint_verified_at"], "2026-08-05T20:00:00+00:00"
            )

    def test_an_unknown_model_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text(json.dumps({"models": []}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                VERIFY.apply_to_registry(path, self.REPORT)


if __name__ == "__main__":
    unittest.main()
