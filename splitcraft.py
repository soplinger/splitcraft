#!/usr/bin/env python3
"""splitcraft: deterministic, stratified, group-aware dataset splits.

Offline and standard-library only. Reads a plain-text sample list, splits it
into train/validation/test (or any named ratio set) with exact integer
accounting, and writes a manifest that can be re-checked later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

MANIFEST_VERSION = 1
MAX_FIELDS = 3
NONE_LABEL = "(none)"
ID_RE = re.compile(r"^[A-Za-z0-9._:/-]+$")
SPLIT_NAMES = {2: ("train", "test"), 3: ("train", "val", "test")}
WARN_DEVIATION_POINTS = 5.0

# Knuth's MMIX linear congruential constants, written in decimal so the
# generator is identical on every platform and Python version.
LCG_MULTIPLIER = 6364136223846793005
LCG_INCREMENT = 1442695040888963407
LCG_MASK = (1 << 64) - 1


class InputError(Exception):
    """Bad input file, manifest or arguments. Reported with exit code 2."""


@dataclass
class Sample:
    id: str
    line: int
    group: str | None
    label: str | None


@dataclass
class Unit:
    """Something that must land in one split: a single sample or a whole group."""

    key: str
    samples: list[Sample]
    stratum: str

    @property
    def weight(self) -> int:
        return len(self.samples)


class LCG:
    """Tiny deterministic generator used only for shuffling."""

    def __init__(self, seed: int):
        self.state = seed & LCG_MASK

    def next_value(self) -> int:
        self.state = (self.state * LCG_MULTIPLIER + LCG_INCREMENT) & LCG_MASK
        return self.state >> 33

    def shuffle(self, items: list) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.next_value() % (i + 1)
            items[i], items[j] = items[j], items[i]


def _field(fields: list[str], index: int) -> str | None:
    if index >= len(fields):
        return None
    value = fields[index]
    return None if value in ("", "-") else value


def parse_sample_text(text: str, path: str = "<input>") -> tuple[list[Sample], list[str]]:
    """Parse `id, group, label` lines. Returns (samples, errors)."""
    samples: list[Sample] = []
    errors: list[str] = []
    first_seen: dict[str, int] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) > MAX_FIELDS:
            errors.append(f"{path}:{lineno}: too many fields ({len(fields)}, max {MAX_FIELDS})")
            continue
        sample_id = fields[0]
        if not sample_id:
            errors.append(f"{path}:{lineno}: missing id")
            continue
        if not ID_RE.match(sample_id):
            errors.append(f"{path}:{lineno}: invalid id {sample_id!r} (use letters, digits, . _ : / -)")
            continue
        if sample_id in first_seen:
            errors.append(
                f"{path}:{lineno}: duplicate id {sample_id!r} (first on line {first_seen[sample_id]})"
            )
            continue
        first_seen[sample_id] = lineno
        samples.append(Sample(sample_id, lineno, _field(fields, 1), _field(fields, 2)))
    return samples, errors


def load_source(path: str) -> tuple[list[Sample], str]:
    """Read and validate a sample file. Returns (samples, sha256 of the bytes)."""
    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(f"{path}: not valid UTF-8 ({exc.reason})")
    samples, errors = parse_sample_text(text, path)
    if errors:
        raise InputError("\n".join(errors))
    if not samples:
        raise InputError(f"{path}: no samples found")
    return samples, hashlib.sha256(data).hexdigest()


def parse_ratios(spec: str) -> dict[str, Fraction]:
    """Turn '70:15:15' into exact, normalised fractions keyed by split name."""
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) < 2:
        raise InputError("ratios need at least two parts, for example 70:15:15")
    values = []
    for part in parts:
        try:
            value = Fraction(part)
        except (ValueError, ZeroDivisionError):
            raise InputError(f"bad ratio part {part!r}")
        if value <= 0:
            raise InputError(f"ratio parts must be positive, got {part!r}")
        values.append(value)
    total = sum(values)
    names = SPLIT_NAMES.get(len(values)) or tuple(f"split{i + 1}" for i in range(len(values)))
    return {name: value / total for name, value in zip(names, values)}


def build_units(samples: list[Sample], by_group: bool) -> list[Unit]:
    """Group samples into indivisible units, each with one stratum label."""
    buckets: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        if by_group and sample.group is not None:
            buckets[f"group:{sample.group}"].append(sample)
        else:
            buckets[f"id:{sample.id}"].append(sample)
    units = []
    for key in sorted(buckets):
        members = buckets[key]
        labels = Counter(m.label for m in members if m.label is not None)
        if labels:
            top = max(labels.values())
            stratum = min(label for label, count in labels.items() if count == top)
        else:
            stratum = NONE_LABEL
        units.append(Unit(key, members, stratum))
    return units


def assign(units: list[Unit], ratios: dict[str, Fraction], seed: int) -> dict[str, str]:
    """Stratified greedy split. Returns {sample id: split name}."""
    rng = LCG(seed)
    names = list(ratios)
    by_stratum: dict[str, list[Unit]] = defaultdict(list)
    for unit in units:
        by_stratum[unit.stratum].append(unit)

    assignments: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        bucket = by_stratum[stratum]
        rng.shuffle(bucket)
        bucket.sort(key=lambda u: -u.weight)  # big groups first; stable, so shuffle breaks ties
        total = sum(u.weight for u in bucket)
        targets = {name: total * ratios[name] for name in names}
        filled = {name: 0 for name in names}
        for unit in bucket:
            best = max(names, key=lambda n: (targets[n] - filled[n], -names.index(n)))
            filled[best] += unit.weight
            for sample in unit.samples:
                assignments[sample.id] = best
    return assignments


def run_split(samples: list[Sample], ratios: dict[str, Fraction], seed: int, by_group: bool) -> dict[str, str]:
    return assign(build_units(samples, by_group), ratios, seed)


def leaked_groups(samples: list[Sample], assignments: dict[str, str]) -> list[str]:
    """Groups whose samples are spread across more than one split."""
    where: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.group is not None and sample.id in assignments:
            where[sample.group].add(assignments[sample.id])
    return sorted(group for group, splits in where.items() if len(splits) > 1)


def render_table(headers: list[str], rows: list[list]) -> list[str]:
    cells = [[str(c) for c in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        parts = [cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths))]
        return "  ".join(parts).rstrip()

    return [fmt(cells[0]), "  ".join("-" * w for w in widths)] + [fmt(r) for r in cells[1:]]


def summarize(
    samples: list[Sample],
    assignments: dict[str, str],
    ratios: dict[str, Fraction],
    by_group: bool,
) -> tuple[list[str], list[str]]:
    """Return (report lines, warnings) for a finished split."""
    names = list(ratios)
    total = len(samples)
    counts = {n: 0 for n in names}
    labels: dict[str, Counter] = {n: Counter() for n in names}
    groups: dict[str, set[str]] = {n: set() for n in names}
    for sample in samples:
        split = assignments[sample.id]
        counts[split] += 1
        labels[split][sample.label or NONE_LABEL] += 1
        if sample.group is not None:
            groups[split].add(sample.group)

    warnings: list[str] = []
    rows = []
    for name in names:
        share = 100 * counts[name] / total
        target = float(ratios[name]) * 100
        rows.append([name, counts[name], f"{share:.1f}%", f"{target:.1f}%", len(groups[name])])
        if counts[name] == 0:
            warnings.append(f"split {name!r} is empty")
        elif abs(share - target) > WARN_DEVIATION_POINTS:
            warnings.append(
                f"split {name!r} holds {share:.1f}% of samples against a {target:.1f}% target "
                "(uneven group sizes or very few samples)"
            )
    lines = ["Splits", *render_table(["split", "samples", "share", "target", "groups"], rows)]

    all_labels = Counter(s.label or NONE_LABEL for s in samples)
    balance_rows = []
    for label in sorted(all_labels):
        balance_rows.append([label, *[labels[n][label] for n in names], all_labels[label]])
        if all_labels[label] < len(names):
            warnings.append(f"label {label!r} has {all_labels[label]} sample(s), fewer than the {len(names)} splits")
        else:
            for name in names:
                if labels[name][label] == 0:
                    warnings.append(f"label {label!r} has no samples in split {name!r}")
    lines += ["", "Class balance", *render_table(["label", *names, "total"], balance_rows)]

    leaks = leaked_groups(samples, assignments)
    if leaks and not by_group:
        warnings.append(
            f"{len(leaks)} group(s) span more than one split, e.g. {leaks[0]!r}; rerun with --by-group"
        )
    return lines, warnings


def write_id_lists(out_dir: str, samples: list[Sample], assignments: dict[str, str], names: list[str]) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        ids = [s.id for s in samples if assignments[s.id] == name]
        (target / f"{name}.txt").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")


def load_manifest(path: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{path}:{exc.lineno}: invalid JSON ({exc.msg})")
    if not isinstance(data, dict) or data.get("tool") != "splitcraft":
        raise InputError(f"{path}: not a splitcraft manifest")
    if data.get("version") != MANIFEST_VERSION:
        raise InputError(f"{path}: unsupported manifest version {data.get('version')!r}")
    required = ("seed", "by_group", "ratios", "source_sha256", "counts", "assignments")
    missing = [key for key in required if key not in data]
    if missing:
        raise InputError(f"{path}: manifest missing {', '.join(missing)}")
    return data


def build_manifest(
    seed: int, by_group: bool, ratios: dict[str, Fraction], sha: str, assignments: dict[str, str]
) -> dict:
    counts = Counter(assignments.values())
    return {
        "tool": "splitcraft",
        "version": MANIFEST_VERSION,
        "seed": seed,
        "by_group": by_group,
        "ratios": {name: str(value) for name, value in ratios.items()},
        "source_sha256": sha,
        "counts": {name: counts.get(name, 0) for name in ratios},
        "assignments": dict(sorted(assignments.items())),
    }


def cmd_split(args: argparse.Namespace) -> int:
    samples, sha = load_source(args.source)
    ratios = parse_ratios(args.ratios)
    assignments = run_split(samples, ratios, args.seed, args.by_group)
    lines, warnings = summarize(samples, assignments, ratios, args.by_group)
    header = f"splitcraft: {len(samples)} samples, seed {args.seed}, ratios {args.ratios}"
    print(header)
    print(f"source sha256 {sha}")
    print()
    print("\n".join(lines))
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if args.manifest:
        manifest = build_manifest(args.seed, args.by_group, ratios, sha, assignments)
        Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"\nManifest written to {args.manifest}")
    if args.out_dir:
        write_id_lists(args.out_dir, samples, assignments, list(ratios))
        print(f"Id lists written to {args.out_dir}/")
    if args.strict and warnings:
        return 1
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    samples, sha = load_source(args.source)
    manifest = load_manifest(args.manifest)
    problems: list[str] = []
    notes: list[str] = []

    if manifest["source_sha256"] != sha:
        problems.append("source file changed since the manifest was written (sha256 differs)")

    source_ids = {s.id for s in samples}
    stored = manifest["assignments"]
    missing = sorted(source_ids - set(stored))
    extra = sorted(set(stored) - source_ids)
    if missing:
        problems.append(f"{len(missing)} source id(s) missing from manifest, e.g. {missing[0]!r}")
    if extra:
        problems.append(f"{len(extra)} manifest id(s) not in source, e.g. {extra[0]!r}")

    ratios = {name: Fraction(value) for name, value in manifest["ratios"].items()}
    bad_splits = sorted(set(stored.values()) - set(ratios))
    if bad_splits:
        problems.append(f"manifest assigns unknown split(s): {', '.join(bad_splits)}")

    if not missing and not extra:
        rebuilt = run_split(samples, ratios, manifest["seed"], manifest["by_group"])
        if rebuilt != stored:
            problems.append("assignments do not reproduce from the stored seed and ratios")
        recount = Counter(stored.values())
        for name in ratios:
            if manifest["counts"].get(name) != recount.get(name, 0):
                problems.append(
                    f"stored count for {name!r} is {manifest['counts'].get(name)}, "
                    f"but the assignments hold {recount.get(name, 0)}"
                )
        leaks = leaked_groups(samples, stored)
        if leaks:
            message = f"{len(leaks)} group(s) span more than one split, e.g. {leaks[0]!r}"
            (problems if manifest["by_group"] else notes).append(message)

    print(f"splitcraft check: {args.source} against {args.manifest}")
    for note in notes:
        print(f"  note: {note}")
    if problems:
        for problem in problems:
            print(f"  FAIL: {problem}")
        return 1
    shares = ", ".join(f"{name} {value}" for name, value in ratios.items())
    print(f"  OK: {len(samples)} samples reproduce from seed {manifest['seed']} ({shares})")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    old = load_manifest(args.old)["assignments"]
    new = load_manifest(args.new)["assignments"]
    common = sorted(set(old) & set(new))
    moved = [i for i in common if old[i] != new[i]]
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    kept = len(common) - len(moved)
    stability = f"{100 * kept / len(common):.1f}%" if common else "n/a"
    print(f"common ids {len(common)}, kept split {kept} ({stability}), moved {len(moved)}, "
          f"added {len(added)}, removed {len(removed)}")

    transitions = Counter((old[i], new[i]) for i in moved)
    if transitions:
        print("\nMoves")
        for (src, dst), count in sorted(transitions.items()):
            print(f"  {src} -> {dst}: {count}")
        print(f"\nFirst {min(args.show, len(moved))} moved id(s)")
        for sample_id in moved[: args.show]:
            print(f"  {sample_id}: {old[sample_id]} -> {new[sample_id]}")
    if added:
        print(f"\nAdded: {', '.join(added[: args.show])}" + (" ..." if len(added) > args.show else ""))
    if removed:
        print(f"Removed: {', '.join(removed[: args.show])}" + (" ..." if len(removed) > args.show else ""))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    samples, sha = load_source(args.source)
    total = len(samples)
    labels = Counter(s.label or NONE_LABEL for s in samples)
    rows = [[label, count, f"{100 * count / total:.1f}%"] for label, count in sorted(labels.items())]
    group_sizes = Counter(s.group for s in samples if s.group is not None)
    print(f"splitcraft stats: {args.source}")
    print(f"  samples {total}, labels {len(labels)}, groups {len(group_sizes)}, "
          f"ungrouped {total - sum(group_sizes.values())}")
    print(f"  source sha256 {sha}")
    if group_sizes:
        sizes = list(group_sizes.values())
        print(f"  group size min {min(sizes)}, median {statistics.median(sizes):g}, max {max(sizes)}")
    print()
    print("\n".join(render_table(["label", "samples", "share"], rows)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splitcraft",
        description="Deterministic, stratified, group-aware dataset splits from plain text.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("split", help="split a sample file into named ratio groups")
    split.add_argument("source", help="sample file: one 'id, group, label' per line")
    split.add_argument("--ratios", default="70:15:15", help="ratio spec, e.g. 70:15:15 or 80:20 (default 70:15:15)")
    split.add_argument("--seed", type=int, default=0, help="shuffle seed (default 0)")
    split.add_argument("--by-group", action="store_true", help="keep every group inside one split")
    split.add_argument("--manifest", metavar="PATH", help="write a reproducible JSON manifest")
    split.add_argument("--out-dir", metavar="DIR", help="write one id list per split")
    split.add_argument("--strict", action="store_true", help="exit 1 if any warning is raised")
    split.set_defaults(func=cmd_split)

    check = sub.add_parser("check", help="verify a manifest against its source file")
    check.add_argument("source")
    check.add_argument("manifest")
    check.set_defaults(func=cmd_check)

    diff = sub.add_parser("diff", help="show which samples moved between two manifests")
    diff.add_argument("old")
    diff.add_argument("new")
    diff.add_argument("--show", type=int, default=10, help="how many ids to list (default 10)")
    diff.set_defaults(func=cmd_diff)

    stats = sub.add_parser("stats", help="label and group summary for a sample file")
    stats.add_argument("source")
    stats.set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except InputError as exc:
        print(f"splitcraft: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"splitcraft: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
