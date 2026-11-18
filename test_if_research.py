import json
import unittest

import numpy as np
import pandas as pd

from experiment import ROOT
from if_research import crossfit_scores, paired_changes


class ResearchTest(unittest.TestCase):
    def test_crossfit_excludes_every_scored_row(self):
        events = pd.DataFrame({"feature": np.arange(25)}, index=np.arange(100, 125))
        seen = []

        class RecordingForest:
            def fit(self, frame):
                self.fitted = set(frame.index)
                return self

            def score_samples(self, frame):
                self_test.assertTrue(self.fitted.isdisjoint(frame.index))
                seen.extend(frame.index)
                return -frame["feature"].to_numpy()

        self_test = self
        values, folds = crossfit_scores(events, (256, 1), 42, factory=lambda *_: RecordingForest())
        np.testing.assert_array_equal(values, events["feature"])
        self.assertEqual(sorted(seen), list(events.index))
        self.assertEqual(len(np.unique(folds)), 5)

    def test_paired_counts_exclude_legitimate_events(self):
        labels = pd.Series([1, 1, 0, 1], index=[10, 20, 30, 40])
        result = paired_changes(labels, {10, 30}, {20, 30, 40})
        self.assertEqual(result["rescued_ids"], [20, 40])
        self.assertEqual(result["regressed_ids"], [10])

    def test_report_reconciles_every_seed_and_keeps_original_reference(self):
        report = json.loads((ROOT / "if-research-report.json").read_text(encoding="utf-8"))
        reference = json.loads((ROOT / "kaggle-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["results"]["original_if_42"]["metrics"],
                         reference["results"]["holdout"]["methods"]["isolation_forest"])
        self.assertEqual(report["results"]["logistic"]["metrics"],
                         reference["results"]["holdout"]["methods"]["logistic"])
        baseline = report["results"]["logistic"]["arrival"]
        for seed in (42, 43, 44):
            name = f"combined_{seed}"
            outcome = report["results"][name]["arrival"]
            changes = report["paired_arrival_changes"][name + "_vs_logistic"]
            self.assertEqual(outcome["caught"] - baseline["caught"],
                             changes["rescued"] - changes["regressed"])
        for result in report["results"].values():
            arrival = result["arrival"]
            self.assertEqual(arrival["caught"] + arrival["missed"], report["evaluation_fraud"])
            self.assertEqual(arrival["caught"] + arrival["false_alerts"], arrival["reviewed"])
            self.assertTrue(all(batch["reviewed"] <= 50 for batch in arrival["batches"]))
        self.assertEqual(report["crossfit"]["rows_scored_once"], report["train_rows"])
        self.assertEqual(len(report["case_groups"]["original_false_alerts"]), 20)
        self.assertEqual(len(report["case_groups"]["original_missed_frauds"]), 20)

    def test_selection_uses_dev_and_the_reported_recommendation_obeys_gate(self):
        report = json.loads((ROOT / "if-research-report.json").read_text(encoding="utf-8"))
        winner = min(report["selection"]["grid"],
                     key=lambda x: (-x["mean_dev_caught"], x["std_dev_caught"],
                                    x["config"]["max_samples"], x["config"]["max_features"]))
        self.assertEqual(winner["config"], report["selection"]["chosen_config"])
        baseline = report["results"]["logistic"]["arrival"]
        eligible = all(
            report["results"][f"combined_{seed}"]["arrival"]["caught"] > baseline["caught"] and
            report["results"][f"combined_{seed}"]["arrival"]["false_alerts"] <= baseline["false_alerts"]
            for seed in (42, 43, 44)
        )
        self.assertEqual(report["recommendation"], "combined_candidate" if eligible else "keep_logistic")


if __name__ == "__main__":
    unittest.main()
