"""Guard the hand-maintained namespace `ehq full`/`ehq pilot` build for `_run`.

`_preset_run` does not pass its own parsed arguments through. It constructs a
fresh `argparse.Namespace` listing each field explicitly, so a new option on the
preset parsers reaches `_run` only if someone also adds it to that list. Missing
one is silent when the reader tolerates absence: `--exclude-model` was accepted
on the command line, never forwarded, and read as "exclude nothing", so a run
that was supposed to withhold a model from the confirmatory analyses instead
published a correlation the exclusion existed to suppress.

These tests compare what `_run` reads against what `_preset_run` forwards, and
against what the preset parsers accept, so the three cannot drift apart again.
"""

import argparse
import ast
import unittest
from pathlib import Path

from ehq.cli import build_parser


SOURCE = (Path(__file__).resolve().parents[1] / "src" / "ehq" / "cli.py").read_text(
    encoding="utf-8"
)
TREE = ast.parse(SOURCE)


def _function(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in cli.py")


def _args_attributes_read_by(function):
    return {
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }


def _forwarded_fields():
    for node in ast.walk(_function("_preset_run")):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Namespace"
        ):
            return {keyword.arg for keyword in node.keywords}
    raise AssertionError("_preset_run no longer builds an argparse.Namespace")


class ForwardingContractTests(unittest.TestCase):
    def test_every_field_run_reads_is_forwarded(self):
        missing = _args_attributes_read_by(_function("_run")) - _forwarded_fields()
        self.assertEqual(
            missing,
            set(),
            f"_preset_run does not forward: {sorted(missing)}. A field _run "
            "reads but never receives is silently absent at runtime.",
        )

    def test_exclude_model_specifically_survives_the_hop(self):
        # The regression that motivated this file.
        self.assertIn("exclude_model", _args_attributes_read_by(_function("_run")))
        self.assertIn("exclude_model", _forwarded_fields())

    def test_run_reads_the_field_directly_rather_than_defaulting(self):
        # `getattr(args, "exclude_model", None) or []` turns a forwarding bug
        # into a valid-looking run. Reading the attribute makes it raise.
        source = ast.get_source_segment(SOURCE, _function("_run"))
        self.assertIn("args.exclude_model", source)
        self.assertNotIn('getattr(args, "exclude_model"', source)
        self.assertNotIn("getattr(args, 'exclude_model'", source)


class ParserAcceptanceTests(unittest.TestCase):
    def _parse(self, study, extra):
        return build_parser().parse_args([study, "--model", "M", *extra])

    def test_both_presets_accept_the_flag(self):
        for study in ("full", "pilot"):
            parsed = self._parse(study, ["--exclude-model", "A=reason"])
            self.assertEqual(parsed.exclude_model, ["A=reason"])

    def test_the_direct_run_commands_accept_it_too(self):
        for study in ("run", "dry-run"):
            parsed = self._parse(study, ["--exclude-model", "A=reason"])
            self.assertEqual(parsed.exclude_model, ["A=reason"])

    def test_it_is_repeatable_and_defaults_to_empty(self):
        parsed = self._parse("full", ["--exclude-model", "A=x", "--exclude-model", "B=y"])
        self.assertEqual(parsed.exclude_model, ["A=x", "B=y"])
        self.assertEqual(self._parse("full", []).exclude_model, [])


if __name__ == "__main__":
    unittest.main()
