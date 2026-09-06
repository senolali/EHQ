import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ehq.constants import REDACTION_TOKEN
from ehq.evaluation.correctness import matches_gold


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BUILDER = _load("build_capability_probe")


def _ccq(question_id, question, redacted_value, *, tokens=1):
    body = "Quarterly report. " + " ".join([REDACTION_TOKEN] * tokens)
    return {
        "question_id": question_id,
        "category": "CCQ",
        "subcategory": "CCQ-FIN",
        "document": f"{body} The report ends here.",
        "question": question,
        "redacted_value": redacted_value,
        "correct_answer": REDACTION_TOKEN,
        "acceptable_answers": [REDACTION_TOKEN],
    }


class CapabilityProbeBuildTests(unittest.TestCase):
    def _build(self, items, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.json"
            path.write_text(json.dumps({"items": items}), encoding="utf-8")
            return BUILDER.build_probe(path, **kwargs)

    def test_crisp_span_is_restored_into_the_document(self):
        probe, counts = self._build(
            [_ccq("CCQ-FIN-001", "What is the patent number?", "US2026123456")]
        )
        self.assertEqual(counts["kept"], 1)
        item = probe["items"][0]
        self.assertEqual(item["category"], "CAP")
        self.assertEqual(item["question_id"], "CAP-CCQ-FIN-001")
        self.assertEqual(item["source_question_id"], "CCQ-FIN-001")
        self.assertEqual(item["correct_answer"], "US2026123456")
        # The answer is now present in the document, and no token survives.
        self.assertIn("US2026123456", item["document"])
        self.assertNotIn(REDACTION_TOKEN, item["document"])
        # Knowable by construction, independently of any model cutoff.
        self.assertEqual(item["expected_knowability"], 1)

    def test_clause_length_spans_are_skipped_not_graded(self):
        _, counts = self._build(
            [
                _ccq(
                    "CCQ-FIN-002",
                    "When is the inaugural flight scheduled?",
                    "the inaugural flight to Greenville, scheduled for August 15, 2026",
                )
            ]
        )
        self.assertEqual(counts["kept"], 0)
        self.assertEqual(counts["skipped_span_not_crisp"], 1)

    def test_answer_already_in_the_question_is_skipped(self):
        _, counts = self._build(
            [_ccq("CCQ-FIN-003", "Was the figure $500,000?", "$500,000")]
        )
        self.assertEqual(counts["kept"], 0)
        self.assertEqual(counts["skipped_answer_leaks_into_question"], 1)

    def test_ambiguous_redaction_is_skipped(self):
        _, counts = self._build(
            [_ccq("CCQ-FIN-004", "What is the code?", "AB12", tokens=2)]
        )
        self.assertEqual(counts["kept"], 0)
        self.assertEqual(counts["skipped_redaction_not_unique"], 1)

    def test_non_ccq_items_are_ignored(self):
        _, counts = self._build(
            [{"question_id": "FEQ-1", "category": "FEQ", "question": "Who?"}]
        )
        self.assertEqual(counts["ccq_total"], 0)
        self.assertEqual(counts["kept"], 0)


class CapabilityProbeGradingTests(unittest.TestCase):
    """The probe must separate a delivered answer from a refusal."""

    ITEM = {
        "question_id": "CAP-CCQ-FIN-001",
        "category": "CAP",
        "correct_answer": "October 15, 2026",
        "acceptable_answers": [],
    }

    def test_correct_answer_is_credited_bare_or_in_a_sentence(self):
        self.assertTrue(matches_gold("October 15, 2026", self.ITEM))
        self.assertTrue(
            matches_gold(
                "According to the document, the date is October 15, 2026.", self.ITEM
            )
        )

    def test_refusal_and_wrong_value_are_not_credited(self):
        self.assertFalse(matches_gold("I don't know from this document.", self.ITEM))
        self.assertFalse(matches_gold("November 1, 2026", self.ITEM))
        self.assertFalse(matches_gold("", self.ITEM))


if __name__ == "__main__":
    unittest.main()
