import re
import unittest
from pathlib import Path

import ehq
from ehq.constants import FRAMEWORK_VERSION, PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_versions_are_single_release_identity(self):
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        version = re.search(r'^version = "([^"]+)"$', project, re.MULTILINE)
        self.assertIsNotNone(version)
        self.assertEqual(version.group(1), FRAMEWORK_VERSION)
        self.assertEqual(ehq.__version__, FRAMEWORK_VERSION)
        self.assertIn(f"version: {FRAMEWORK_VERSION}", cff)
        self.assertEqual(PROTOCOL_VERSION, "1.1.4")

    def test_distribution_name_and_console_entry_point(self):
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertRegex(project, r'(?m)^name = "ehq"$')
        self.assertRegex(project, r'(?m)^ehq = "ehq\.cli:main"$')


if __name__ == "__main__":
    unittest.main()
