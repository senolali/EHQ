import unittest

from ehq.datasets.pcq_sources import assign_fact_ids, parse_weo_table_rows


class PCQSourceTests(unittest.TestCase):
    def test_weo_parser_ignores_numeric_header_and_normalizes_minus(self):
        text = """
                         2024 2025 2026 2027 2026 2027 2025 2026 2027
        World Output      3.5  3.5  3.0  3.4  –0.1 0.2  3.4  2.9  3.4
        United States     2.8  2.1  2.3  2.2   0.0 0.1  2.0  2.3  2.1
        """
        rows = parse_weo_table_rows(text)
        self.assertEqual([row["label"] for row in rows], ["World Output", "United States"])
        self.assertEqual(rows[0]["values"][4], "-0.1")

    def test_fact_ids_are_stable_and_mark_automatic_verification(self):
        facts = [
            {
                "subcategory": "PCQ-X",
                "event_date": "2026-07-01",
                "question_target": "first target",
                "gold_answer": "alpha",
                "source_evidence": [],
            },
            {
                "subcategory": "PCQ-X",
                "event_date": "2026-07-02",
                "question_target": "second target",
                "gold_answer": "beta",
                "source_evidence": [],
            },
        ]
        assigned = assign_fact_ids(facts, subcategory="PCQ-X")
        self.assertEqual(
            [fact["fact_id"] for fact in assigned],
            ["PCQ-X-0001", "PCQ-X-0002"],
        )
        self.assertTrue(
            all(
                fact["verification"]["status"] == "source-verified"
                and fact["verification"]["reviewer_type"]
                == "automated-source-check"
                for fact in assigned
            )
        )
