"""Regression checks for zero-fit repairs and safe incremental reruns.

Run without pytest: python -m unittest discover -s tests -p test_minimal_rerun.py
"""

import json
import io
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline
from evaluation.revision_analysis import explanation_metrics
from evaluation.temporal_cv import ObservableTimeSeriesSplit
from repair_reports import repair_reports
from rerun_support import descriptive_ranks, load_checkpoint, save_checkpoint, training_key


def toy_data():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.normal(size=100), "cat": rng.integers(0, 3, 100)})
    y = pd.Series(np.tile([0, 0, 0, 0, 1], 20), index=X.index)
    return dict(X_train=X, X_test=X.iloc[:20], y_train=y, y_test=y.iloc[:20],
                prep_spec={"continuous_columns": ["a"], "indicator_columns": [], "nominal_columns": ["cat"]},
                nominal_idx=[1], cat_dims=[4], cutoff=None, train_dates=None, split_mode="random")


def arguments():
    return SimpleNamespace(dataset="uci", split_mode="random", seed=42, fast=True, tune=False,
                           tune_iter=1, inner_folds=3, inner_temporal_folds=3,
                           strategy_selection="per_model", models=["Logistic Reg."],
                           skip_tabnet=True, resume=True, evaluate_only=True)


class MinimalRerunTests(unittest.TestCase):
    def test_reversed_rank_agreement_stays_negative(self):
        result = explanation_metrics(np.array([[1., 2., 3.]]),
                                     [pd.Series({"a": 3., "b": 2., "c": 1.})], ["a", "b", "c"])
        self.assertEqual(result.shap_lime_abs_rank_rho.iloc[0], -1.)

    def test_month_folds_only_use_available_labels(self):
        origination = pd.Series(np.repeat(pd.date_range("2005-01-01", periods=48, freq="MS"), 20))
        dates = pd.DataFrame({"_origination": origination,
                              "_label_observable_by": origination + pd.DateOffset(months=12)})
        X = pd.DataFrame({"x": np.arange(len(dates))})
        y = np.tile([0, 0, 0, 0, 1], len(dates) // 5)
        folds = list(ObservableTimeSeriesSplit(dates, 3).split(X, y))
        self.assertEqual(len(folds), 3)
        for train, validation in folds:
            start = origination.iloc[validation].min()
            self.assertTrue((dates.iloc[train]._label_observable_by <= start).all())
            self.assertFalse(set(origination.iloc[train].dt.to_period("M")) &
                             set(origination.iloc[validation].dt.to_period("M")))
        changed = X.copy(); changed.index = changed.index + 1
        with self.assertRaisesRegex(ValueError, "not aligned"):
            list(ObservableTimeSeriesSplit(dates, 3).split(changed, y))

    def test_checkpoint_rejects_stale_training_inputs(self):
        data, args = toy_data(), arguments()
        key = training_key(args, data)
        args.threshold_criterion = "cost"
        self.assertEqual(key, training_key(args, data))
        args.tune_iter = 2
        different = training_key(args, data)
        self.assertNotEqual(key, different)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pkl"
            save_checkpoint(path, key, {"finished": True})
            self.assertEqual(load_checkpoint(path, key), {"finished": True})
            self.assertIsNone(load_checkpoint(path, different))
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_evaluate_only_reuses_complete_pipeline_without_fitting(self):
        data, args = toy_data(), arguments()
        data["cache_key"] = training_key(args, data)
        model = pipeline._make_pipeline("None", "Logistic Reg.", 42, data)
        model.fit(data["X_train"], data["y_train"])
        predictions = model.predict_proba(data["X_train"])[:, 1]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(pipeline, "MDL_DIR", Path(directory)), \
             patch.object(pipeline, "RES_DIR", Path(directory)):
            save_checkpoint(Path(directory) / "logistic_reg._training.pkl", data["cache_key"], {
                "pipeline": model, "record": {"status": "success"}, "oof": predictions,
                "mask": np.ones(len(predictions), dtype=bool), "strategy_choice": {"strategy": "None"},
            })
            with patch.object(pipeline, "_oof", side_effect=AssertionError("Unexpected fitting")), \
                 patch.object(pipeline, "_make_pipeline", side_effect=AssertionError("Unexpected model construction")):
                result = pipeline.stage_training(data, {"ablation_df": pd.DataFrame({"Strategy": ["None"]}),
                                                        "best_strategy": "None"}, args)
            np.testing.assert_allclose(result["trained_models"]["Logistic Reg."].predict_proba(data["X_train"])[:, 1], predictions)

    def test_missing_evaluation_checkpoint_never_falls_back_to_training(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "MDL_DIR", Path(directory)):
            with patch.object(pipeline, "_oof", side_effect=AssertionError("Unexpected fitting")):
                with self.assertRaisesRegex(RuntimeError, "No matching imbalance checkpoint"):
                    pipeline.stage_imbalance_ablation({"cache_key": "missing"}, arguments())
                with self.assertRaisesRegex(RuntimeError, "never fits models"):
                    pipeline.stage_training({"cache_key": "missing"},
                                            {"ablation_df": pd.DataFrame({"Strategy": ["None"]})}, arguments())

    def test_identical_oof_recipe_is_fitted_only_once(self):
        data, args = toy_data(), arguments()
        data["cache_key"] = training_key(args, data)
        expected = (np.linspace(0., 1., len(data["y_train"])), np.ones(len(data["y_train"]), dtype=bool))
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "MDL_DIR", Path(directory)):
            with patch.object(pipeline, "oof_predict_proba", return_value=expected) as fit:
                for _ in range(2):
                    model = pipeline._make_pipeline("None", "Logistic Reg.", 42, data)
                    result = pipeline._oof(model, data, args)
                    np.testing.assert_array_equal(result[0], expected[0])
                self.assertEqual(fit.call_count, 1)
                data["cache_key"] = "changed training inputs"
                pipeline._oof(model, data, args)
                self.assertEqual(fit.call_count, 2)

    def test_report_repair_preserves_source_and_removes_independence_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "fixed"
            run = source / "uci/runs/seed_42/results"; run.mkdir(parents=True)
            frame = pd.DataFrame({"instance": [0, 1], "shap_lime_abs_rank_rho": [1., .5],
                                  "shap_lime_signed_abs_attribution_rho": [-1., .5]})
            path = run / "explanation_quality_metrics.csv"; frame.to_csv(path, index=False)
            summary = source / "uci/runs/summary"; summary.mkdir()
            (summary / "friedman_nemenyi_across_runs.json").write_text('{"p_value":0.001}')
            pd.DataFrame({"run_id": ["seed_42", "seed_42"], "model": ["a", "b"],
                          "AUROC": [.8, .7]}).to_csv(summary / "all_run_test_metrics.csv", index=False)
            original = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            info = repair_reports(source, target)
            self.assertEqual(info["model_fits"], 0); self.assertEqual(info["agreement_rows_changed"], 1)
            self.assertEqual(pd.read_csv(target / path.relative_to(source)).shap_lime_abs_rank_rho.iloc[0], -1.)
            self.assertFalse((target / "uci/runs/summary/friedman_nemenyi_across_runs.json").exists())
            self.assertTrue((target / "uci/runs/summary/descriptive_model_ranks.json").exists())
            self.assertEqual(original, {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()})
            with self.assertRaises(FileExistsError):
                repair_reports(source, target)

    def test_temporal_seed_repetitions_are_not_independent_rank_blocks(self):
        frame = pd.DataFrame({"run_id": ["a", "b", "c", "d"], "cutoff": ["2011-07"] * 4,
                              "model": ["x", "y", "x", "y"], "AUROC": [.8, .7, .82, .72]})
        result = descriptive_ranks(frame)
        self.assertEqual(result["metrics"]["AUROC"]["n_blocks"], 1)
        self.assertNotIn("p_value", json.dumps(result))

    def test_legacy_scalar_record_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_params.yaml"
            path.write_text('models:\n  Random Forest:\n    imbalance_strategy: "None"\n'
                            '    best_params:\n      max_depth: null\n      n_estimators: 200\n'
                            '    status: "success"\n  CatBoost:\n    status: "disabled"\n')
            record = pipeline._legacy_record(path, "Random Forest")
            self.assertEqual(record["best_params"], {"max_depth": None, "n_estimators": 200})
            self.assertEqual(record["imbalance_strategy"], "None")

    def test_tabnet_scheduler_class_can_be_checkpointed(self):
        from models.tabnet_model import _ReduceLROnPlateau
        self.assertIs(pickle.loads(pickle.dumps(_ReduceLROnPlateau)), _ReduceLROnPlateau)

    def test_selective_summary_keeps_compatible_runs_and_omits_stale_reports(self):
        metric = {name: .7 for name in ("AUROC", "AUPRC", "F1", "Precision", "Recall", "Specificity", "G-Mean", "MCC", "Brier")}
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "RUNS_DIR", Path(directory)):
            root = Path(directory)
            template = dict(dataset="uci", split_mode="random", dataset_checksums={},
                            source_fingerprint="unchanged", package_versions={}, skip_ablation_grid=True,
                            revision_analyses=False, bootstrap_reps=0, cost_fp=1., protocol=pipeline.PROTOCOL)
            saved = dict(results={"a": metric}, oof_results={"a": metric}, imbalance={}, tuning_records={})
            for seed, cost in ((42, 1.), (99, 1.), (123, 2.)):
                target = root / f"seed_{seed}/results"; target.mkdir(parents=True)
                manifest = dict(template, seed=seed, cutoff=None, run_id=f"seed_{seed}", cost_fp=cost, results_dir=str(target))
                pipeline._write_json(manifest, target / "run_manifest.json")
                pipeline._write_json(saved, target / "run_summary.json")
                pd.DataFrame({"obsolete": [1]}).to_csv(target / "bootstrap_metric_confidence_intervals.csv", index=False)
                pd.DataFrame({"obsolete": [1]}).to_csv(target / "smote_model_ablation.csv", index=False)
                if seed == 42:
                    current = dict(run_id=manifest["run_id"], seed=seed, cutoff=None, manifest=manifest,
                                   dirs={"res_dir": target}, eval=saved, imbalance={"smote_summary": {}},
                                   train={"tuning_records": {}})
            summary = root / "summary"; summary.mkdir()
            stale = summary / "all_run_bootstrap_metric_confidence_intervals.csv"; stale.write_text("obsolete\n1\n")
            pipeline._write_multi_run_summary([current])
            frame = pd.read_csv(summary / "all_run_test_metrics.csv")
            self.assertEqual(set(frame.run_id), {"seed_42", "seed_99"})
            self.assertFalse(stale.exists())
            self.assertFalse((summary / "all_run_model_smote_ablation.csv").exists())

    def test_active_output_defaults_and_archive_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active, archive = root / "outputs", root / "outputs_original_archive"
            archive.mkdir()
            with patch.object(pipeline, "OUTPUTS_DIR", active), \
                 patch.object(pipeline, "ORIGINAL_OUTPUTS_DIR", archive), \
                 patch.object(pipeline, "RUNS_DIR", root), \
                 patch.object(pipeline, "_run_one", return_value={"run_id": "seed_42"}) as run, \
                 patch.object(pipeline, "_write_multi_run_summary"), \
                 patch.object(sys, "stderr", io.StringIO()):
                with patch.object(sys, "argv", ["pipeline.py", "--seeds", "42"]):
                    pipeline.main()
                args = run.call_args.args[0]
                self.assertEqual(args.out_dir, active)
                self.assertEqual(args.reuse_models_from, archive)
                run.reset_mock()
                for protected in (archive, archive / "child", root / "outputs_v2_archive"):
                    with patch.object(sys, "argv", ["pipeline.py", "--out-dir", str(protected)]):
                        with self.assertRaises(SystemExit) as exc:
                            pipeline.main()
                        self.assertEqual(exc.exception.code, 2)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
