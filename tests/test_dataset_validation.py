import unittest

from ehq.datasets.validate import validate_dataset


def _ccq(question_id, question="What value was redacted?"):
    return {
        "question_id": question_id,
        "category": "CCQ",
        "subcategory": "CCQ-TEST",
        "document": "The audited value was [REDACTED] for the quarter.",
        "question": question,
        "redacted_value": "42 units",
        "correct_answer": "[REDACTED]",
        "qc_passed": True,
    }


class DatasetValidationTests(unittest.TestCase):
    def test_valid_single_ccq_item(self):
        self.assertTrue(validate_dataset([_ccq("CCQ-001")]).valid)

    def test_repeated_ccq_content_fails_release_gate(self):
        report = validate_dataset([_ccq("CCQ-001"), _ccq("CCQ-002")])
        codes = [issue.code for issue in report.issues]
        self.assertFalse(report.valid)
        self.assertIn("duplicate_question", codes)
        self.assertIn("duplicate_content", codes)

    def test_ccq_answer_leak_is_detected(self):
        item = _ccq("CCQ-001")
        item["document"] += " The value 42 units was reviewed."
        report = validate_dataset([item])
        self.assertTrue(any(issue.code == "ccq_answer_leak" for issue in report.issues))

    def test_short_fact_does_not_match_inside_unrelated_words(self):
        item = _ccq("CCQ-001")
        item["redacted_value"] = "O+"
        item["document"] = (
            "The blood type was [REDACTED]. "
            "A routine follow-up confirmed no complications."
        )
        report = validate_dataset([item])
        self.assertFalse(
            any(issue.code == "ccq_answer_leak" for issue in report.issues)
        )

    def test_balanced_release_requirement(self):
        report = validate_dataset(
            [_ccq("CCQ-001")],
            expected_total=3000,
            expected_per_category=750,
        )
        self.assertFalse(report.valid)
        self.assertTrue(any(issue.code == "unexpected_total" for issue in report.issues))
        self.assertEqual(
            sum(issue.code == "category_imbalance" for issue in report.issues), 4
        )

    def test_declared_candidate_gate_requires_explicit_opt_in(self):
        item = {
            "question_id": "PCQ-001",
            "category": "PCQ",
            "subcategory": "PCQ-TEST",
            "question": "Which verified event occurred after the cutoff date?",
            "correct_answer": "Event A",
            "release_status": "candidate_requires_human_review",
            "qc": {
                "passed": False,
                "checks": {
                    "automatic": {"passed": True},
                    "human_source_review": {
                        "passed": False,
                        "status": "mandatory_review_pending",
                    },
                },
            },
        }
        self.assertFalse(validate_dataset([item]).valid)
        allowed = validate_dataset([item], allow_pending_human_review=True)
        self.assertTrue(allowed.valid)
        self.assertEqual(allowed.issues[0].severity, "warning")

    def test_candidate_mode_does_not_hide_unknown_qc_failure(self):
        item = _ccq("CCQ-001")
        item["release_status"] = "candidate_requires_human_review"
        item["qc"] = {
            "passed": False,
            "checks": {"novel_failure": {"passed": False}},
        }
        report = validate_dataset([item], allow_pending_human_review=True)
        self.assertFalse(report.valid)
        self.assertTrue(any(issue.code == "qc_failed" for issue in report.issues))

    def test_completed_primary_review_can_retain_secondary_review_gate(self):
        item = {
            "question_id": "HNQ-001",
            "category": "HNQ",
            "subcategory": "HNQ-TEST",
            "question": "Which obscure historical result is documented here?",
            "correct_answer": "Result A",
            "release_status": "candidate_requires_human_review",
            "qc": {
                "passed": False,
                "checks": {
                    "human_source_review": {
                        "passed": True,
                        "status": "completed",
                        "reviewer_name": "Primary Reviewer",
                    },
                    "independent_secondary_review": {
                        "passed": False,
                        "status": "required_independent_review_pending",
                        "reviewer_must_differ_from_primary": True,
                        "fixed_stratified_sample_size": 300,
                    },
                },
            },
        }
        self.assertFalse(validate_dataset([item]).valid)
        allowed = validate_dataset([item], allow_pending_human_review=True)
        self.assertTrue(allowed.valid, allowed.to_dict())
        self.assertEqual(allowed.issues[0].code, "candidate_human_gate_pending")

    def test_release_pcq_requires_human_verified_claim_temporality(self):
        item = {
            "question_id": "PCQ-POL-X",
            "category": "PCQ",
            "subcategory": "PCQ-POL",
            "question": "Who was appointed to the new office in July 2026?",
            "correct_answer": "Example Person",
            "event_date": "2026-07-08",
            "source_evidence": [
                {
                    "source_id": "OFFICIAL-1",
                    "url": "https://example.test/appointment",
                    "publisher": "Example Authority",
                    "verified_at": "2026-07-09",
                    "evidence": "Example Person was appointed on 2026-07-08.",
                }
            ],
            "qc_passed": True,
        }
        missing = validate_dataset(
            [item],
            require_source_evidence=True,
            require_pcq_temporal_novelty=True,
        )
        self.assertFalse(missing.valid)
        self.assertTrue(
            any(issue.code == "missing_temporal_novelty" for issue in missing.issues)
        )

        item["temporal_novelty"] = {
            "status": "human-verified-post-cutoff",
            "basis": "official_appointment",
            "claim_became_true_at": "2026-07-08",
            "source_id": "OFFICIAL-1",
            "reviewer_type": "named-human-reviewer",
            "verified_at": "2026-07-09",
            "evidence": "The appointment itself occurred on 2026-07-08.",
        }
        accepted = validate_dataset(
            [item],
            require_source_evidence=True,
            require_pcq_temporal_novelty=True,
        )
        self.assertTrue(accepted.valid, accepted.to_dict())

    def test_source_publication_is_not_an_allowed_temporal_basis(self):
        item = {
            "question_id": "PCQ-POL-HIST",
            "category": "PCQ",
            "subcategory": "PCQ-POL",
            "question": "In which year did Türkiye join NATO?",
            "correct_answer": "1952",
            "event_date": "2026-07-08",
            "source_evidence": [
                {
                    "source_id": "NATO-OVERVIEW",
                    "url": "https://example.test/nato",
                    "publisher": "NATO",
                    "verified_at": "2026-07-09",
                    "evidence": "Türkiye joined NATO in 1952.",
                }
            ],
            "temporal_novelty": {
                "status": "human-verified-post-cutoff",
                "basis": "source_publication_date",
                "claim_became_true_at": "2026-07-08",
                "source_id": "NATO-OVERVIEW",
                "reviewer_type": "named-human-reviewer",
                "verified_at": "2026-07-09",
                "evidence": "The page was published in 2026.",
            },
            "qc_passed": True,
        }
        report = validate_dataset(
            [item], require_pcq_temporal_novelty=True
        )
        self.assertFalse(report.valid)
        self.assertTrue(
            any(
                issue.code == "invalid_temporal_novelty_basis"
                for issue in report.issues
            )
        )
