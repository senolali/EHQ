import unittest

from ehq.datasets.ledger import validate_fact_ledger


def _fact(fact_id="PCQ-ECO-1", target="What was the projection?", gold="1.7%"):
    return {
        "fact_id": fact_id,
        "subcategory": "PCQ-ECO",
        "event_date": "2026-07-01",
        "question_target": target,
        "gold_answer": gold,
        "source_evidence": [
            {
                "source_id": "S",
                "url": "https://example.test",
                "publisher": "Example",
                "verified_at": "2026-07-02",
                "evidence": "The projection was published.",
            }
        ],
    }


class LedgerTests(unittest.TestCase):
    def test_balanced_minimal_ledger_is_structurally_valid(self):
        ledger = {
            "event_window": {"start": "2026-06-01", "end": "2026-07-25"},
            "facts": [_fact()],
        }
        report = validate_fact_ledger(ledger, expected_total=1)
        self.assertTrue(report["valid"], report["issues"])

    def test_duplicate_atomic_fact_and_target_fail(self):
        fact = _fact()
        ledger = {
            "event_window": {"start": "2026-06-01", "end": "2026-07-25"},
            "facts": [fact, {**fact, "fact_id": "PCQ-ECO-2"}],
        }
        report = validate_fact_ledger(ledger)
        codes = [issue["code"] for issue in report["issues"]]
        self.assertFalse(report["valid"])
        self.assertIn("duplicate_atomic_fact", codes)
        self.assertIn("duplicate_question_target", codes)

    def test_same_target_with_different_gold_is_a_hard_conflict(self):
        ledger = {
            "event_window": {"start": "2026-06-01", "end": "2026-07-25"},
            "facts": [
                _fact(),
                _fact(
                    fact_id="PCQ-ECO-2",
                    target="what was the projection",
                    gold="3.4%",
                ),
            ],
        }
        report = validate_fact_ledger(ledger)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertFalse(report["valid"])
        self.assertIn("duplicate_question_target", codes)
        self.assertIn("conflicting_question_target_gold", codes)

    def test_expected_subcategories_include_missing_groups(self):
        ledger = {
            "event_window": {"start": "2026-06-01", "end": "2026-07-25"},
            "facts": [_fact()],
        }
        report = validate_fact_ledger(
            ledger,
            expected_per_subcategory=1,
            expected_subcategories=("PCQ-ECO", "PCQ-POL"),
        )
        self.assertFalse(report["valid"])
        self.assertTrue(
            any(
                issue["code"] == "subcategory_imbalance"
                and "PCQ-POL" in issue["message"]
                for issue in report["issues"]
            )
        )

    def test_release_evidence_requires_snapshot_and_verification(self):
        ledger = {
            "event_window": {"start": "2026-06-01", "end": "2026-07-25"},
            "facts": [_fact()],
        }
        report = validate_fact_ledger(ledger, require_release_evidence=True)
        self.assertFalse(report["valid"])
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("incomplete_release_evidence", codes)
        self.assertIn("missing_verification", codes)
        self.assertIn("missing_temporal_novelty", codes)


if __name__ == "__main__":
    unittest.main()
