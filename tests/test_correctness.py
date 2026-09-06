import unittest

from ehq.evaluation.correctness import matches_gold, normalize_text


class CorrectnessTests(unittest.TestCase):
    def test_unicode_and_punctuation_normalization(self):
        self.assertEqual(normalize_text("İstanbul—Türkiye"), "istanbul turkiye")

    def test_matches_explicit_gold_inside_explanation(self):
        item = {
            "category": "PCQ",
            "correct_answer": "Ferran Torres",
            "acceptable_answers": ["Torres"],
        }
        self.assertTrue(matches_gold("The winning scorer was Ferran Torres.", item))
        self.assertFalse(matches_gold("The winning scorer was Rodri.", item))

    def test_feq_and_ccq_never_treat_substantive_claim_as_gold(self):
        self.assertFalse(
            matches_gold(
                "Elara Voss wrote the paper.",
                {"category": "FEQ", "correct_answer": "[DOES_NOT_EXIST]"},
            )
        )
        self.assertFalse(
            matches_gold(
                "470,000 shares",
                {"category": "CCQ", "correct_answer": "[REDACTED]"},
            )
        )

    def test_numeric_matching_is_strict_by_default(self):
        item = {"category": "HNQ", "correct_answer": "100.0"}
        self.assertTrue(matches_gold("100", item))
        self.assertFalse(matches_gold("104", item))
        self.assertTrue(matches_gold("104", item, numeric_tolerance=0.05))
