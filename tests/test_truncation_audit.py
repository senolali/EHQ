import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_truncated_responses", ROOT / "tools" / "audit_truncated_responses.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record(model, question_id, text, label="CONFIDENT_WRONG", tokens=None):
    envelope = {"ok": True, "text": text}
    if tokens is not None:
        envelope["usage"] = {"completion_tokens": tokens}
    return {
        "model": model,
        "question_id": question_id,
        "answer_response": envelope,
        "classification": {"label": label},
    }


def _audit(rows, limit=512):
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "records.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        return MODULE.audit(run, answer_max_tokens=limit)


class StrongSignalTests(unittest.TestCase):
    def test_answer_at_the_token_budget_is_flagged(self):
        stats = _audit(
            [_record("M", "Q1", "The identifier is US202612 and the value", tokens=512)]
        )["models"]["M"]
        self.assertEqual(stats["token_limit"], 1)
        self.assertEqual(stats["suspected"], 1)
        self.assertEqual(dict(stats["labels_of_suspected"]), {"CONFIDENT_WRONG": 1})

    def test_clause_stopping_on_a_function_word_is_flagged(self):
        for tail in (
            "Based only on the information present, the identifier assigned to",
            "Based on the document, the new flagship store opened on",
            "The value of the primary asset of the company is",
        ):
            with self.subTest(tail=tail):
                self.assertEqual(_audit([_record("M", "Q", tail)])["models"]["M"]["suspected"], 1)

    def test_dangling_separator_is_flagged(self):
        stats = _audit(
            [_record("M", "Q1", "The patient's last colonoscopy was on March 22,")]
        )["models"]["M"]
        self.assertEqual(stats["incomplete_clause"], 1)

    def test_unclosed_quote_is_flagged(self):
        stats = _audit(
            [_record("M", "Q1", "Based on the document, the identifier is 'US202612")]
        )["models"]["M"]
        self.assertEqual(stats["incomplete_clause"], 1)


class FalsePositiveTests(unittest.TestCase):
    """A complete answer must not be reported as cut off."""

    def test_complete_sentence_without_a_full_stop_is_not_suspected(self):
        # Observed on LLaMA-3-70B: a finished statement, merely unpunctuated.
        stats = _audit(
            [
                _record(
                    "M",
                    "Q1",
                    "The Sociedad Argentina para la Conservacion de la Fauna "
                    "was founded in 1967",
                    tokens=19,
                )
            ]
        )["models"]["M"]
        self.assertEqual(stats["suspected"], 0)
        # Still reported under the weak signal for the panel comparison.
        self.assertEqual(stats["unpunctuated"], 1)

    def test_bare_value_answer_is_not_suspected(self):
        stats = _audit(
            [_record("M", "Q1", "US2026123456", label="CONFIDENT_CORRECT", tokens=8)]
        )["models"]["M"]
        self.assertEqual(stats["suspected"], 0)

    def test_complete_prose_answer_is_not_suspected(self):
        stats = _audit(
            [_record("M", "Q1", "The answer is US2026123456.", tokens=20)]
        )["models"]["M"]
        self.assertEqual(stats["suspected"], 0)


class PanelComparisonTests(unittest.TestCase):
    def test_a_model_far_above_its_peers_is_marked_an_outlier(self):
        rows = []
        # Twelve peers answer cleanly.
        for index in range(12):
            rows.append(_record(f"Peer{index}", "Q1", "The answer is 42.", tokens=10))
        # One model leaves almost everything unpunctuated without any single
        # record proving truncation on its own.
        for index in range(10):
            rows.append(
                _record("Suspect", f"Q{index}", "The flight to Greenville South Carolina")
            )
        report = _audit(rows)
        self.assertTrue(report["models"]["Suspect"]["panel_outlier"])
        self.assertFalse(report["models"]["Peer0"]["panel_outlier"])

    def test_uniformly_tidy_panel_produces_no_alarm(self):
        rows = [
            _record(f"M{index}", "Q1", "The answer is 42.", tokens=10)
            for index in range(5)
        ]
        report = _audit(rows)
        self.assertFalse(any(s["panel_outlier"] for s in report["models"].values()))


class RecordShapeTests(unittest.TestCase):
    def test_probe_shaped_records_are_read(self):
        report = _audit(
            [{"model": "M", "question_id": "Q1", "response_text": "Value is US20261"}]
        )
        self.assertEqual(report["models"]["M"]["n_answers"], 1)


if __name__ == "__main__":
    unittest.main()


class ApostropheTests(unittest.TestCase):
    """Possessives and contractions must not read as an unclosed quotation."""

    def test_possessive_and_quoted_title_are_not_suspected(self):
        stats = _audit(
            [_record("M", "Q1", "Martha Graham's solo 'Scherza' premiered in 1944")]
        )["models"]["M"]
        self.assertEqual(stats["suspected"], 0)

    def test_unclosed_double_quote_is_still_flagged(self):
        stats = _audit(
            [_record("M", "Q1", 'The document states that "Dr. Hart was appointed')]
        )["models"]["M"]
        self.assertEqual(stats["incomplete_clause"], 1)
