import unittest

from ehq.datasets.ccq_audit import audit_ccq


def _item(question_id, question, document, value):
    return {
        "question_id": question_id,
        "category": "CCQ",
        "subcategory": "CCQ-TEST",
        "question": question,
        "document": document,
        "redacted_value": value,
        "correct_answer": "[REDACTED]",
        "provenance": {
            "origin": "synthetic-ccq",
            "generator": {
                "requested_model": "mistral-large",
                "resolved_model": "mistral-large",
                "cache_hit": False,
            },
        },
        "qc": {"passed": True},
    }


class CCQAuditTests(unittest.TestCase):
    def test_reports_near_duplicates_and_attempt_cost(self):
        left = _item(
            "CCQ-TEST-001",
            "Which control value was removed from the quarterly report?",
            "The fictional quarterly report contains [REDACTED] and routine context.",
            "ZX-1",
        )
        right = _item(
            "CCQ-TEST-002",
            "Which control value was removed from this quarterly report?",
            "This fictional quarterly report contains [REDACTED] and routine context.",
            "ZX-2",
        )
        attempts = [
            {
                "question_id": left["question_id"],
                "accepted": True,
                "item": {
                    **left,
                    "provenance": {
                        **left["provenance"],
                        "generator": {
                            **left["provenance"]["generator"],
                            "usage": {
                                "total_token_count": 100,
                                "total_token_cost": "0.01",
                            },
                        },
                    },
                },
            }
        ]
        report = audit_ccq(
            [left, right],
            attempts=attempts,
            expected_per_subcategory=2,
            question_similarity_threshold=0.70,
            document_similarity_threshold=0.70,
        )
        self.assertFalse(report["automatic_gate_passed"])
        self.assertTrue(report["near_duplicate_questions"])
        self.assertTrue(report["near_duplicate_documents"])
        self.assertEqual(report["attempts"]["total_tokens"], 100)
        self.assertEqual(report["attempts"]["reported_total_cost"], "0.01")

    def test_nested_qc_failure_is_not_structurally_valid(self):
        item = _item(
            "CCQ-TEST-001",
            "Which value was removed from this fictional report?",
            "The fictional report contains [REDACTED] and routine context.",
            "ZX-1",
        )
        item["qc"]["passed"] = False
        report = audit_ccq([item])
        self.assertFalse(report["validation"]["valid"])
