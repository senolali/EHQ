import random
import unittest

from ehq.selection import select_items, selection_manifest


class SelectionTests(unittest.TestCase):
    def test_selection_is_order_independent_and_balanced(self):
        items = [
            {
                "question_id": f"{category}-{subcategory}-{index:02d}",
                "category": category,
                "subcategory": f"{category}-{subcategory}",
            }
            for category in ("FEQ", "PCQ", "HNQ", "CCQ")
            for subcategory in ("A", "B")
            for index in range(10)
        ]
        shuffled = list(items)
        random.Random(99).shuffle(shuffled)
        first = select_items(items, limit=16, seed=42)
        second = select_items(shuffled, limit=16, seed=42)
        self.assertEqual(
            [item["question_id"] for item in first],
            [item["question_id"] for item in second],
        )
        report = selection_manifest(first)
        self.assertEqual(set(report["stratum_counts"].values()), {2})

    def test_category_filter_is_strict(self):
        items = [
            {"question_id": "FEQ-1", "category": "FEQ", "subcategory": "F"},
            {"question_id": "PCQ-1", "category": "PCQ", "subcategory": "P"},
        ]
        selected = select_items(items, categories=["pcq"], seed=1)
        self.assertEqual([row["question_id"] for row in selected], ["PCQ-1"])
        with self.assertRaises(ValueError):
            select_items(items, categories=["OTHER"])
