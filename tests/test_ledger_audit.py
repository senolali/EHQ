import unittest

from ehq.datasets.ledger_audit import audit_ledger_dataset
from ehq.hashing import sha256_json


def _fact(fact_id, subcategory, gold):
    return {
        "fact_id": fact_id,
        "subcategory": subcategory,
        "event_date": "2026-07-01",
        "question_target": f"target for {fact_id}",
        "gold_answer": gold,
        "source_evidence": [
            {
                "source_id": f"S-{fact_id}",
                "url": "https://example.test/source",
                "publisher": "Example",
                "verified_at": "2026-07-02",
                "evidence": f"The supported answer is {gold}.",
            }
        ],
    }


def _item(fact, question):
    return {
        "schema_version": "1.0",
        "question_id": fact["fact_id"],
        "category": "PCQ",
        "subcategory": fact["subcategory"],
        "question": question,
        "correct_answer": fact["gold_answer"],
        "acceptable_answers": [],
        "expected_knowability": 0,
        "event_date": fact["event_date"],
        "source_evidence": fact["source_evidence"],
        "provenance": {
            "origin": "source-ledger-pcq",
            "ledger_fact_sha256": sha256_json(fact),
            "generator": {
                "name": "asu",
                "requested_model": "mistral-large",
                "resolved_model": "mistral-large",
            },
        },
        "qc": {"passed": True, "checks": {}, "notes": []},
    }


class LedgerAuditTests(unittest.TestCase):
    def test_balanced_source_grounded_dataset_passes(self):
        facts = [
            _fact("PCQ-A-001", "PCQ-A", "Alpha"),
            _fact("PCQ-B-001", "PCQ-B", "Beta"),
        ]
        report = audit_ledger_dataset(
            [
                _item(facts[0], "Which result was recorded for the first event?"),
                _item(facts[1], "What outcome was announced for the second event?"),
            ],
            ledger={
                "event_window": {"start": "2026-06-01", "end": "2026-07-25"},
                "facts": facts,
            },
            category="PCQ",
            subcategories=("PCQ-A", "PCQ-B"),
            expected_per_subcategory=1,
        )
        self.assertTrue(report["automatic_gate_passed"], report)

    def test_hash_mismatch_and_near_duplicate_fail(self):
        facts = [
            _fact("PCQ-A-001", "PCQ-A", "Alpha"),
            _fact("PCQ-B-001", "PCQ-B", "Beta"),
        ]
        items = [
            _item(facts[0], "Which result was recorded for the tournament final?"),
            _item(facts[1], "Which result was recorded for the tournament final?"),
        ]
        items[1]["provenance"]["ledger_fact_sha256"] = "0" * 64
        report = audit_ledger_dataset(
            items,
            ledger={
                "event_window": {"start": "2026-06-01", "end": "2026-07-25"},
                "facts": facts,
            },
            category="PCQ",
            subcategories=("PCQ-A", "PCQ-B"),
            expected_per_subcategory=1,
        )
        self.assertFalse(report["automatic_gate_passed"])
        self.assertTrue(report["ledger_mismatches"])
        self.assertTrue(report["near_duplicate_questions"])
