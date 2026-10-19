import contextlib
import io
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import splitcraft as sc  # noqa: E402


def make_samples(text):
    samples, errors = sc.parse_sample_text(text)
    assert not errors, errors
    return samples


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = sc.main(argv)
    return code, out.getvalue(), err.getvalue()


class ParseTests(unittest.TestCase):
    def test_comments_blank_lines_and_optional_fields(self):
        samples, errors = sc.parse_sample_text("a\n# note\n\nb, g1\nc, -, cat\n")
        self.assertEqual(errors, [])
        self.assertEqual([s.id for s in samples], ["a", "b", "c"])
        self.assertIsNone(samples[0].group)
        self.assertEqual(samples[1].group, "g1")
        self.assertIsNone(samples[2].group)
        self.assertEqual(samples[2].label, "cat")
        self.assertEqual(samples[2].line, 5)

    def test_duplicate_id_reports_both_lines(self):
        _, errors = sc.parse_sample_text("x\ny\nx\n", "data.txt")
        self.assertEqual(len(errors), 1)
        self.assertIn("data.txt:3:", errors[0])
        self.assertIn("first on line 1", errors[0])

    def test_too_many_fields_is_line_numbered(self):
        _, errors = sc.parse_sample_text("a\na,b,c,d\n", "f")
        self.assertEqual(errors, ["f:2: too many fields (4, max 3)"])

    def test_invalid_id_characters_are_rejected(self):
        _, errors = sc.parse_sample_text("bad id\n")
        self.assertEqual(len(errors), 1)
        self.assertIn("invalid id 'bad id'", errors[0])

    def test_missing_id_is_rejected(self):
        _, errors = sc.parse_sample_text(", g1, cat\n")
        self.assertEqual(len(errors), 1)
        self.assertIn("missing id", errors[0])


class RatioTests(unittest.TestCase):
    def test_three_part_ratio_is_exact_and_normalised(self):
        ratios = sc.parse_ratios("70:15:15")
        self.assertEqual(list(ratios), ["train", "val", "test"])
        self.assertEqual(ratios["train"], Fraction(7, 10))
        self.assertEqual(ratios["val"], Fraction(3, 20))
        self.assertEqual(sum(ratios.values()), 1)

    def test_two_part_ratio_uses_train_and_test(self):
        ratios = sc.parse_ratios("80:20")
        self.assertEqual(list(ratios), ["train", "test"])
        self.assertEqual(ratios["train"], Fraction(4, 5))

    def test_bad_ratios_are_input_errors(self):
        for spec in ("70", "0:1", "x:1", "1:-2"):
            with self.subTest(spec=spec):
                with self.assertRaises(sc.InputError):
                    sc.parse_ratios(spec)


class AssignmentTests(unittest.TestCase):
    def test_unlabelled_split_gives_exact_counts(self):
        samples = make_samples("".join(f"s{i}\n" for i in range(10)))
        assignments = sc.run_split(samples, sc.parse_ratios("7:3"), seed=5, by_group=False)
        counts = {name: list(assignments.values()).count(name) for name in ("train", "test")}
        self.assertEqual(counts, {"train": 7, "test": 3})

    def test_stratification_keeps_label_ratio_in_each_split(self):
        lines = "".join(f"c{i}, -, cat\n" for i in range(8)) + "d0, -, dog\nd1, -, dog\n"
        samples = make_samples(lines)
        assignments = sc.run_split(samples, sc.parse_ratios("1:1"), seed=3, by_group=False)
        for name in ("train", "test"):
            labels = [s.label for s in samples if assignments[s.id] == name]
            self.assertEqual(labels.count("cat"), 4)
            self.assertEqual(labels.count("dog"), 1)

    def test_by_group_never_splits_a_group(self):
        lines = "".join(f"s{i}, g{i // 2}, cat\n" for i in range(40))
        samples = make_samples(lines)
        assignments = sc.run_split(samples, sc.parse_ratios("70:15:15"), seed=11, by_group=True)
        self.assertEqual(sc.leaked_groups(samples, assignments), [])
        self.assertEqual(len(assignments), 40)

    def test_every_sample_assigned_exactly_once(self):
        samples = make_samples("a, g1, x\nb, g1, y\nc, g2, x\nd, -, y\ne, g3, x\n")
        assignments = sc.run_split(samples, sc.parse_ratios("2:1"), seed=0, by_group=True)
        self.assertEqual(set(assignments), {"a", "b", "c", "d", "e"})
        self.assertEqual(set(assignments.values()), {"train", "test"})

    def test_input_order_does_not_change_result(self):
        samples = make_samples("".join(f"s{i}, g{i % 5}, L{i % 3}\n" for i in range(30)))
        forward = sc.run_split(samples, sc.parse_ratios("70:30"), seed=9, by_group=True)
        backward = sc.run_split(list(reversed(samples)), sc.parse_ratios("70:30"), seed=9, by_group=True)
        self.assertEqual(forward, backward)

    def test_mixed_label_group_takes_majority_then_alphabetical_stratum(self):
        samples = make_samples("a, g, cat\nb, g, cat\nc, g, dog\n")
        units = sc.build_units(samples, by_group=True)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].stratum, "cat")
        self.assertEqual(units[0].weight, 3)
        tied = make_samples("a, g, dog\nb, g, cat\n")
        self.assertEqual(sc.build_units(tied, by_group=True)[0].stratum, "cat")

    def test_lcg_is_deterministic_per_seed(self):
        first = [sc.LCG(42).next_value() for _ in range(5)]
        again = [sc.LCG(42).next_value() for _ in range(5)]
        other = [sc.LCG(43).next_value() for _ in range(5)]
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)


class LeakageTests(unittest.TestCase):
    def test_group_spanning_two_splits_is_reported(self):
        samples = make_samples("a, g, x\nb, g, x\n")
        self.assertEqual(sc.leaked_groups(samples, {"a": "train", "b": "test"}), ["g"])
        self.assertEqual(sc.leaked_groups(samples, {"a": "train", "b": "train"}), [])


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        lines = "".join(f"s{i:02d}, g{i // 2}, {'cat' if i % 3 else 'dog'}\n" for i in range(24))
        self.source = self.dir / "samples.txt"
        self.source.write_text(lines, encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_split_check_round_trip_passes(self):
        manifest = self.dir / "m.json"
        code, out, _ = run_cli(["split", str(self.source), "--by-group", "--seed", "4", "--manifest", str(manifest)])
        self.assertEqual(code, 0)
        self.assertIn("Class balance", out)
        code, out, _ = run_cli(["check", str(self.source), str(manifest)])
        self.assertEqual(code, 0, out)
        self.assertIn("OK: 24 samples", out)

    def test_tampered_manifest_fails_check(self):
        manifest = self.dir / "m.json"
        run_cli(["split", str(self.source), "--seed", "4", "--manifest", str(manifest)])
        data = json.loads(manifest.read_text(encoding="utf-8"))
        first = next(iter(data["assignments"]))
        data["assignments"][first] = "test" if data["assignments"][first] != "test" else "train"
        manifest.write_text(json.dumps(data), encoding="utf-8")
        code, out, _ = run_cli(["check", str(self.source), str(manifest)])
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)

    def test_changed_source_fails_check(self):
        manifest = self.dir / "m.json"
        run_cli(["split", str(self.source), "--seed", "4", "--manifest", str(manifest)])
        self.source.write_text(self.source.read_text(encoding="utf-8") + "extra1\n", encoding="utf-8")
        code, out, _ = run_cli(["check", str(self.source), str(manifest)])
        self.assertEqual(code, 1)
        self.assertIn("sha256 differs", out)

    def test_parse_errors_exit_two_with_line_numbers(self):
        bad = self.dir / "bad.txt"
        bad.write_text("a\na\n", encoding="utf-8")
        code, _, err = run_cli(["split", str(bad)])
        self.assertEqual(code, 2)
        self.assertIn("bad.txt:2: duplicate id 'a'", err)

    def test_out_dir_writes_one_list_per_split(self):
        out_dir = self.dir / "lists"
        code, _, _ = run_cli(["split", str(self.source), "--ratios", "2:1:1", "--out-dir", str(out_dir)])
        self.assertEqual(code, 0)
        ids = []
        for name in ("train", "val", "test"):
            ids += (out_dir / f"{name}.txt").read_text(encoding="utf-8").split()
        self.assertEqual(sorted(ids), sorted(f"s{i:02d}" for i in range(24)))

    def test_strict_mode_fails_on_empty_split(self):
        tiny = self.dir / "tiny.txt"
        tiny.write_text("a\nb\nc\n", encoding="utf-8")
        code, out, _ = run_cli(["split", str(tiny), "--ratios", "1:1:98"])
        self.assertEqual(code, 0)
        self.assertIn("split 'train' is empty", out)
        code, _, _ = run_cli(["split", str(tiny), "--ratios", "1:1:98", "--strict"])
        self.assertEqual(code, 1)

    def test_diff_of_identical_manifests_moves_nothing(self):
        manifest = self.dir / "m.json"
        run_cli(["split", str(self.source), "--seed", "1", "--manifest", str(manifest)])
        code, out, _ = run_cli(["diff", str(manifest), str(manifest)])
        self.assertEqual(code, 0)
        self.assertIn("moved 0", out)
        self.assertIn("(100.0%)", out)

    def test_stats_reports_label_shares(self):
        code, out, _ = run_cli(["stats", str(self.source)])
        self.assertEqual(code, 0)
        self.assertIn("samples 24", out)
        self.assertIn("groups 12", out)


if __name__ == "__main__":
    unittest.main()
