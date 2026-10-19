# splitcraft

**Deterministic, stratified, group-aware dataset splits from one plain-text file.**

splitcraft takes a list of samples, splits it into train, validation and test (or any named ratio set) with exact integer accounting, keeps related rows together, and writes a manifest that anyone can re-run and check later. It runs offline on the Python standard library. No servers, no accounts, no network calls.

## Features

- **Plain-text sample files.** One `id, group, label` line per sample. Group and label are optional, and `-` means "none".
- **Line-numbered errors.** Duplicate ids, bad characters, extra fields and missing ids are all reported with `file:line`, in one pass.
- **Exact ratios.** Ratios such as `70:15:15` or `80:20` are parsed as fractions, so split sizes are never off by floating-point drift.
- **Stratified splits.** Each label keeps the same share in every split, so rare classes are not lost in one partition.
- **Group-aware splits.** `--by-group` keeps every group (patient, camera, session) inside one split, which prevents leakage.
- **Reproducible shuffles.** A small built-in generator with fixed decimal constants gives the same split on every platform. Input line order does not change the result.
- **JSON manifests.** `--manifest` records the seed, ratios, source hash, counts and every assignment.
- **Re-check a manifest.** `check` re-runs the split from the stored seed and ratios, verifies the source hash, and catches tampered or stale assignments. It exits 1 on failure.
- **Diff two manifests.** `diff` reports how many samples kept their split, which moved where, and which were added or removed. Useful when you add data to a dataset.
- **Id lists for pipelines.** `--out-dir` writes one `<split>.txt` per split, in source order.
- **Quality warnings.** Empty splits, labels missing from a split, labels with fewer samples than splits, split shares far from target, and group leakage are all flagged.
- **CI gate.** `--strict` makes warnings fail the build. Exit codes are 0 for success, 1 for a failed check or strict warning, and 2 for input errors.
- **Stats view.** `stats` summarises labels and group sizes before you commit to a split.

## Sample file format

```text
# id, group, label
img001, pat01, cat
img002, pat01, cat
img003, pat02, dog
img032, -, dog
```

- Lines starting with `#` and blank lines are ignored.
- Ids may use letters, digits and `. _ : / -`.
- Trailing fields may be left out. `img032` alone is a valid line.
- With `--by-group`, a sample whose group is `-` is treated as its own unit.
- A group that mixes labels is stratified under its most common label. Ties go to the alphabetically first label.

A complete example is in [`examples/samples.txt`](examples/samples.txt).

## Quick start

Requires Python 3. There is nothing to install.

```sh
# 1. Look at the data: labels, group sizes, sample counts
python3 splitcraft.py stats examples/samples.txt

# 2. Split 70:15:15, keeping patient groups together, and save a manifest and id lists
python3 splitcraft.py split examples/samples.txt --by-group --seed 7 \
    --manifest splits.json --out-dir lists/

# 3. Verify the manifest against the source file (exit 0 when it passes)
python3 splitcraft.py check examples/samples.txt splits.json

# 4. After adding rows to the source, see how many samples kept their split
python3 splitcraft.py split examples/samples.txt --by-group --seed 7 --manifest after.json
python3 splitcraft.py diff splits.json after.json
```

`split` prints the source hash, a **Splits** table (samples, share, target and group count per split), a **Class balance** table (label counts per split), and a **Warnings** section when anything needs a look.

Add `--strict` to fail the build on any warning. Use `--ratios 80:20` for a two-way split, which gives `train` and `test`. Any other number of parts gives `split1`, `split2` and so on.

## How to run

```sh
python3 splitcraft.py split <samples.txt> [--ratios 70:15:15] [--seed N] [--by-group]
                        [--manifest PATH] [--out-dir DIR] [--strict]
python3 splitcraft.py check <samples.txt> <manifest.json>
python3 splitcraft.py diff <old.json> <new.json> [--show N]
python3 splitcraft.py stats <samples.txt>
```

| Exit code | Meaning |
|-----------|---------|
| 0 | Success |
| 1 | `check` failed, or `split --strict` raised a warning |
| 2 | Bad input: parse error, bad ratio, unreadable or invalid manifest |

## How to test

The tests use `unittest` from the standard library, so they run without any extra packages:

```sh
python3 -m unittest discover -s tests -v
```

If you have pytest installed, it collects the same tests:

```sh
python3 -m pytest -q
```

The suite covers parsing and line-numbered errors, exact ratio parsing, exact split sizes, stratification, group integrity, input-order independence, the generator, leakage detection, the CLI round trip, tamper and source-change detection, strict mode, id-list output, diffs and stats.

## Project layout

```text
splitcraft.py                 single-file CLI and library
examples/samples.txt          sample input with groups and labels
tests/test_splitcraft.py      unit and CLI tests
README.md
```

## Notes on the method

- Units are samples, or whole groups under `--by-group`. Units are shuffled inside each label stratum, then the largest units are placed first. Each unit goes to the split with the biggest remaining deficit against its target. This keeps sample shares close to the ratios even when group sizes differ.
- Splits are not guaranteed to match the ratios exactly when groups are large. Check the warnings and the target column.
- The seed alone determines the shuffle, so changing the seed changes the assignment, and so does changing the ratios or `--by-group`.
