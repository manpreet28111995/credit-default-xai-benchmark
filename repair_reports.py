"""Repair saved reporting artifacts without fitting models or explainers.

The source tree is read-only. Corrected tables are written to a separate tree.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from rerun_support import BOOTSTRAP_SCOPE, descriptive_ranks


def repair_reports(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate, non-nested directories.")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}; choose a new output directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance = {"source": str(source), "model_fits": 0, "explainer_runs": 0,
                  "protocol_note": "Training-only CV selection; OOF scores are selection diagnostics, not unbiased estimates.",
                  "bootstrap_scope": BOOTSTRAP_SCOPE,
                  "temporal_note": "Original inner TimeSeriesSplit used retrospective labels; these reporting repairs do not correct temporal model selection.",
                  "source_sha256": {}, "agreement_rows_changed": 0, "agreement_tables": 0}
    with tempfile.TemporaryDirectory(prefix="report-repair-", dir=destination.parent) as temporary:
        output = Path(temporary) / "outputs"
        output.mkdir()
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix not in {".csv", ".json", ".tex", ".yaml"}:
                continue
            if "models" in path.relative_to(source).parts:
                continue
            relative = path.relative_to(source)
            provenance["source_sha256"][str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
            # Preserve original inference in the source tree. The corrected tree
            # reports descriptive ranks because repeated seeds are dependent.
            if path.name == "friedman_nemenyi_across_runs.json":
                continue
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if path.name == "explanation_quality_metrics.csv":
                frame = pd.read_csv(path)
                signed = "shap_lime_signed_abs_attribution_rho"
                if signed not in frame:
                    raise ValueError(f"Cannot recover signed agreement from {path}")
                provenance["agreement_rows_changed"] += int((frame[signed] < 0).sum())
                provenance["agreement_tables"] += 1
                frame["shap_lime_abs_rank_rho"] = frame[signed]
                frame.to_csv(target, index=False)
            elif "bootstrap" in path.name and path.suffix == ".csv":
                try:
                    frame = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    continue
                frame["inference_scope"] = BOOTSTRAP_SCOPE
                frame.to_csv(target, index=False)
        # Regenerate after copying so old .tex files cannot overwrite repairs.
        for path in output.rglob("explanation_quality_metrics.csv"):
            frame = pd.read_csv(path)
            table = frame.describe().round(4).rename_axis("statistic").reset_index()
            latex = table.to_latex(index=False, float_format="%.4f",
                                  caption="Explanation parsimony, stability, and signed SHAP--LIME rank agreement",
                                  label="tab:explanation_quality")
            path.with_name("table5_explanation_quality.tex").write_text(latex)
        for path in output.rglob("all_run_test_metrics.csv"):
            ranks = descriptive_ranks(pd.read_csv(path))
            path.with_name("descriptive_model_ranks.json").write_text(json.dumps(ranks, indent=2))
        (output / "reporting_corrections.json").write_text(json.dumps(provenance, indent=2))
        output.rename(destination)
    return provenance


def main():
    root = Path(__file__).parent / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    archive = root / "outputs_original_archive"
    parser.add_argument("--source", type=Path, default=archive if archive.exists() else root / "outputs")
    parser.add_argument("--out-dir", type=Path, default=root / "reporting_fixed")
    args = parser.parse_args()
    result = repair_reports(args.source, args.out_dir)
    print(f"Repaired {result['agreement_tables']} agreement tables; "
          f"{result['agreement_rows_changed']} signed correlations restored; "
          f"0 model fits. Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
