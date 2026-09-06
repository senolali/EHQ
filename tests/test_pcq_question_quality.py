import unittest

from ehq.datasets.question_quality import pcq_question_quality_issues
from ehq.datasets.validate import validate_dataset


class PCQQuestionQualityTests(unittest.TestCase):
    def test_nested_interrogative_template_is_rejected(self):
        malformed = (
            "In the cited NATO material, what was "
            "What is the goal for defence investment by 2035?"
        )
        self.assertIn(
            "nested_interrogative_template",
            pcq_question_quality_issues(malformed),
        )
        report = validate_dataset(
            [
                {
                    "question_id": "PCQ-POL-X",
                    "category": "PCQ",
                    "subcategory": "PCQ-POL",
                    "question": malformed,
                    "correct_answer": "5%",
                    "qc_passed": True,
                }
            ]
        )
        self.assertFalse(report.valid)
        self.assertTrue(
            any(issue.code == "pcq_malformed_question" for issue in report.issues)
        )

    def test_imperative_source_question_is_allowed(self):
        question = (
            "For the table entry labelled India, report the IMF's "
            "year-over-year figure for 2027."
        )
        self.assertEqual(pcq_question_quality_issues(question), ())

    def test_direct_complete_question_is_allowed(self):
        self.assertEqual(
            pcq_question_quality_issues("Who became the new president of Caltech?"),
            (),
        )


if __name__ == "__main__":
    unittest.main()
