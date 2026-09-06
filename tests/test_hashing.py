from pathlib import Path
import tempfile
import unittest

from ehq.hashing import sha256_tree


class HashingTests(unittest.TestCase):
    def test_tree_hash_tracks_source_content_not_creation_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.py").write_text("b = 1\n", encoding="utf-8")
            (root / "a.py").write_text("a = 1\n", encoding="utf-8")
            first = sha256_tree(root)
            (root / "ignored.txt").write_text("ignored", encoding="utf-8")
            self.assertEqual(first, sha256_tree(root))
            (root / "a.py").write_text("a = 2\n", encoding="utf-8")
            self.assertNotEqual(first, sha256_tree(root))
