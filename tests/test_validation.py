import unittest

from ehq.validation import (
    binary_response_label,
    classification_metrics,
    cohen_kappa,
)


class ValidationMetricTests(unittest.TestCase):
    def test_perfect_kappa(self):
        values = ["ABSTAIN", "HEDGE", "CONFIDENT_WRONG"]
        self.assertEqual(cohen_kappa(values, values), 1.0)

    def test_classification_metrics_include_per_class_results(self):
        reference = ["ABSTAIN", "HEDGE", "CONFIDENT_WRONG", "CONFIDENT_CORRECT"]
        predicted = ["ABSTAIN", "CONFIDENT_WRONG", "CONFIDENT_WRONG", "CONFIDENT_CORRECT"]
        result = classification_metrics(reference, predicted)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["accuracy"], 0.75)
        hedge = next(row for row in result["per_class"] if row["label"] == "HEDGE")
        self.assertEqual(hedge["support"], 1)
        self.assertEqual(hedge["recall"], 0.0)

    def test_binary_mapping(self):
        self.assertEqual(binary_response_label("ABSTAIN"), "RESTRAINT")
        self.assertEqual(binary_response_label("HEDGE"), "RESTRAINT")
        self.assertEqual(
            binary_response_label("CONFIDENT_CORRECT"), "SUBSTANTIVE"
        )


if __name__ == "__main__":
    unittest.main()
