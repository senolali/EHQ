import unittest

from ehq.evaluation.confidence import (
    parse_confidence,
    parse_confidence_detailed,
)


class ConfidenceTests(unittest.TestCase):
    def test_accepts_protocol_integer_scale(self):
        self.assertEqual(parse_confidence("0"), 0.0)
        self.assertEqual(parse_confidence("73"), 0.73)
        self.assertEqual(parse_confidence("100%"), 1.0)

    def test_rejects_ambiguous_or_out_of_range_values(self):
        self.assertIsNone(parse_confidence("8/10"))
        self.assertIsNone(parse_confidence("between 70 and 80"))
        self.assertIsNone(parse_confidence("model v2 confidence 80"))
        self.assertIsNone(parse_confidence("the 85th percentile"))
        self.assertIsNone(parse_confidence("0.85"))
        self.assertIsNone(parse_confidence("101"))
        self.assertIsNone(parse_confidence("-1"))
        self.assertIsNone(parse_confidence(""))
        self.assertIsNone(parse_confidence(None))

    def test_accepts_one_unambiguous_integer_inside_prose(self):
        self.assertEqual(parse_confidence("Confidence: 80"), 0.8)
        self.assertEqual(parse_confidence("I'd say around 85%."), 0.85)
        result = parse_confidence_detailed("Confidence: 80")
        self.assertEqual(result.strategy, "single_integer_in_text")
        self.assertIsNone(result.reason)
