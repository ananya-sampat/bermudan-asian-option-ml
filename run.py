"""One entry point for saved results, checks, and the original experiment suite."""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import runpy
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
for directory in ("archive", "experiments", "src"):
    sys.path.insert(0, str(ROOT / directory))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="summary",
        choices=["summary", "check", "full", "original"],
    )
    parser.add_argument(
        "script", nargs="?", help="Original filename for the original command"
    )
    args = parser.parse_args()
    if args.script and args.command != "original":
        parser.error("A script name is only valid with original.")

    if args.command == "summary":
        with (ROOT / "results_ml/policy_test.csv").open() as f:
            rows = list(csv.DictReader(f))
        print("Saved Asian-policy results: mean across three training seeds")
        print("(Historical measured results; this command does not retrain.)")
        for family in dict.fromkeys(row["family"] for row in rows):
            values = [float(row["mean"]) for row in rows if row["family"] == family]
            print(f"{family:12s} ${sum(values) / len(values):.6f}")
    elif args.command == "check":
        import compare_ml

        # Existing checks must not rewrite the supplied research outputs.
        with tempfile.TemporaryDirectory(prefix="option_ml_checks_") as tmp:
            compare_ml.OUT = Path(tmp)
            compare_ml.checks()
            compare_ml.verify_integration()
        print("Checks passed; saved experiment results were not modified.")
    elif args.command == "full":
        import compare_ml
        output = ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True)
        compare_ml.OUT = output / "results_ml"
        compare_ml.FIG = output / "figures_ml"
        compare_ml.OUT.mkdir()
        compare_ml.FIG.mkdir()
        print(f"Running the full suite. New results: {output}", flush=True)
        compare_ml.run()
        print(f"Completed. Results and figures: {output}")
    else:
        allowed = {
            p.name: p
            for folder in ("src", "archive")
            for p in (ROOT / folder).glob("*.py")
            if p.stem not in {"ml_models", "continuation_dataset", "optimizers"}
        }
        if args.script not in allowed:
            parser.error(
                "Choose an original filename, e.g. asian_compare.py. See docs/FILE_AUDIT.md."
            )
        # Plotting examples write to an isolated directory, not the source tree.
        output = (
            ROOT / "runs" / ("original_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        )
        output.mkdir(parents=True)
        os.chdir(output)
        runpy.run_path(str(allowed[args.script]), run_name="__main__")


if __name__ == "__main__":
    main()
