"""Derived analyses must not be written inside a sealed run directory.

`verify_run_artifacts` treats a file absent from the catalogue as a failure, so
an analysis written into a completed run invalidates it -- and the publication
builder refuses to run against an unverified source. That is the catalogue
doing its job; the fix belongs in the tools, which write beside the run instead.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ("truncation_sensitivity", "abstention_confidence_compliance")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = Path(tempfile.gettempdir()) / "ehq-test" / "outputs" / "real_panel-014"


class OutputLocationTests(unittest.TestCase):
    def test_the_default_is_a_sibling_of_the_run(self):
        for name in TOOLS:
            with self.subTest(tool=name):
                resolved = _load(name).resolve_output_dir(RUN, None)
                self.assertEqual(resolved.name, "real_panel-014_analysis")
                self.assertEqual(resolved.parent, RUN.parent)
                self.assertNotIn(RUN, resolved.parents)

    def test_writing_into_the_run_is_refused(self):
        for name in TOOLS:
            module = _load(name)
            for inside in (RUN, RUN / "sensitivity", RUN / "report" / "tables"):
                with self.subTest(tool=name, path=inside):
                    with self.assertRaises(SystemExit) as caught:
                        module.resolve_output_dir(RUN, inside)
                    self.assertIn("artifact catalogue", str(caught.exception))

    def test_an_explicit_outside_location_is_honoured(self):
        target = Path(tempfile.gettempdir()) / "ehq-test" / "paper" / "tables"
        for name in TOOLS:
            with self.subTest(tool=name):
                self.assertEqual(
                    _load(name).resolve_output_dir(RUN, target), target.resolve()
                )


if __name__ == "__main__":
    unittest.main()
