import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiment import ROOT
from if_feature_study import configuration_key, select_features
from kaggle_benchmark import FEATURES


class FeatureStudyTest(unittest.TestCase):
    def test_relevance_receives_training_labels_outside_feature_matrix(self):
        train = pd.DataFrame(np.ones((8, len(FEATURES))), columns=FEATURES)
        train["Class"] = [0, 0, 0, 1, 0, 0, 0, 0]
        captured = {}

        def relevance(inputs, labels, **kwargs):
            captured["columns"] = list(inputs.columns)
            captured["labels"] = labels.tolist()
            captured["seed"] = kwargs["random_state"]
            return np.arange(len(FEATURES), dtype=float)

        with patch("if_feature_study.mutual_info_classif", side_effect=relevance):
            ranking = select_features(train)
        self.assertEqual(captured["columns"], FEATURES)
        self.assertEqual(captured["labels"], train["Class"].tolist())
        self.assertEqual(captured["seed"], 42)
        self.assertEqual(ranking[0][0], "log_amount")
        self.assertNotIn("Class", [name for name, _ in ranking])

    def test_report_choices_and_training_populations_match_protocol(self):
        report = json.loads((ROOT / "if-feature-report.json").read_text(encoding="utf-8"))
        selection = report["selection"]
        self.assertEqual(selection, json.loads((ROOT / "if-feature-selection.json").read_text(encoding="utf-8")))
        for objective in ("standalone", "secondary"):
            chosen = min(selection["candidates"], key=lambda row: configuration_key(row, objective))
            self.assertEqual(chosen["name"], selection[objective])
        ranked = [row["feature"] for row in selection["train_feature_relevance"]]
        for config in selection["candidates"]:
            expected_rows = report["normal_training_rows"] if config["normal_only"] else report["train_rows"]
            self.assertEqual(config["training_rows"], expected_rows)
            expected_features = FEATURES if len(config["features"]) == 29 else ranked[:len(config["features"])]
            self.assertEqual(config["features"], expected_features)

    def test_evaluation_counts_and_original_baselines_reconcile(self):
        report = json.loads((ROOT / "if-feature-report.json").read_text(encoding="utf-8"))
        old_if = json.loads((ROOT / "if-research-report.json").read_text(encoding="utf-8"))
        old_xgb = json.loads((ROOT / "xgb-queue-report.json").read_text(encoding="utf-8"))
        selected = report["selection"]["standalone"]
        for seed in report["seeds"]:
            for field in ("caught", "false_alerts", "reviewed"):
                self.assertEqual(report["results"][f"all_top29_{seed}"]["standalone"][field],
                                 old_if["results"][f"selected_if_{seed}"]["arrival"][field])
                self.assertEqual(report["results"][f"main_50_{seed}"]["arrival"][field],
                                 old_xgb["results"][f"main_50_{seed}"]["arrival"][field])
            before = report["results"][f"all_top29_{seed}"]["standalone"]["caught"]
            after = report["results"][f"{selected}_{seed}"]["standalone"]["caught"]
            change = report["comparisons"][f"standalone_{seed}"]
            self.assertEqual(after - before, len(change["rescued_ids"]) - len(change["regressed_ids"]))
        for row in report["results"].values():
            outcome = row.get("standalone", row.get("arrival"))
            self.assertEqual(outcome["caught"] + outcome["missed"], 115)
            self.assertEqual(outcome["caught"] + outcome["false_alerts"], outcome["reviewed"])
            self.assertTrue(all(batch["reviewed"] <= 50 for batch in outcome["batches"]))

    def test_unique_if_examples_are_labelled_fraud_and_missed_by_xgb(self):
        report = json.loads((ROOT / "if-feature-report.json").read_text(encoding="utf-8"))
        by_id = {row["row_id"]: row for row in report["cases"]}
        for row in report["cases"]:
            for key, trace in row["traces"].items():
                if key.startswith("reserved"):
                    self.assertIsNotNone(trace["if_score"])
                else:
                    self.assertIsNone(trace["if_score"])
        for seed in report["seeds"]:
            for row_id in report["case_groups"][f"if_unique_catches_{seed}"]:
                row = by_id[row_id]
                self.assertEqual(row["label"], 1)
                self.assertEqual(row["traces"][f"reserved_45_5_{seed}"]["branch"], "if")
                self.assertEqual(row["traces"][f"reserved_45_5_{seed}"]["decision"], "review")
                self.assertNotEqual(row["traces"][f"main_50_{seed}"]["decision"], "review")
        secondary = next(row for row in report["selection"]["candidates"] if row["name"] == report["selection"]["secondary"])
        passes = all(
            run["reserved"]["caught"] > report["selection"]["main_dev"][str(run["seed"])]["caught"] and
            run["reserved"]["false_alerts"] <= report["selection"]["main_dev"][str(run["seed"])]["false_alerts"]
            for run in secondary["dev_runs"]
        )
        self.assertEqual(report["selection"]["policy"], "reserved_45_5" if passes else "main_50")


if __name__ == "__main__":
    unittest.main()
