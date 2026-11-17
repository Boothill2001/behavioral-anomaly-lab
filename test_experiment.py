import unittest
import json

import numpy as np
import pandas as pd

from experiment import ROOT, feature_rows, rank_and_measure
from operations import distribution_shift, fit_cutoff, replay_queue


class ExperimentTest(unittest.TestCase):
    def test_history_is_invariant_to_future_and_same_timestamp_events(self):
        sample = pd.DataFrame({
            "TRANSACTION_ID": [10, 11, 12, 13],
            "CUSTOMER_ID": [2, 2, 2, 2],
            "TERMINAL_ID": [4, 5, 5, 5],
            "TX_DATETIME": pd.to_datetime(["2018-04-01 10:00", "2018-04-01 10:30",
                                           "2018-04-01 10:30", "2018-04-01 11:30"]),
            "TX_AMOUNT": [20, 40, 90, 10],
        })
        full = feature_rows(sample)
        prefix = feature_rows(sample.iloc[:2])
        pd.testing.assert_series_equal(full.iloc[1], prefix.iloc[1])
        self.assertEqual(full["prior_tx_total"].tolist(), [0, 1, 1, 3])
        self.assertEqual(full["prior_tx_1h"].tolist(), [0, 1, 1, 2])
        self.assertEqual(full["new_terminal"].tolist(), [1, 1, 1, 0])
        self.assertEqual(full["amount_ratio_to_history"].iloc[1], 2)

    def test_budget_resets_per_day_and_uses_stable_id_tie_breaks(self):
        ids = list(range(20, 9, -1)) + list(range(31, 20, -1))
        rows = pd.DataFrame({"day": ["a"] * 11 + ["b"] * 11, "TRANSACTION_ID": ids,
                             "TX_FRAUD": [int(i in (20, 21)) for i in ids]})
        ranked, report = rank_and_measure(rows, [1] * len(ids))
        self.assertEqual(ranked.iloc[0]["TRANSACTION_ID"], 10)
        self.assertEqual(report["budgets"][10]["reviewed"], 20)
        self.assertEqual(report["budgets"][10]["tp"], 1)
        self.assertEqual(report["budgets"][10]["fp"], 19)
        self.assertEqual(report["budgets"][10]["fn"], 1)

    def test_arrival_policy_respects_capacity_and_never_uses_labels(self):
        events = pd.DataFrame({"day": [7, 6, 6, 6], "Time": [24, 12, 11, 10],
                               "TRANSACTION_ID": [4, 3, 2, 1], "TX_FRAUD": [1, 1, 1, 0]})
        cutoff = fit_cutoff(np.array([2., 9., 5., 1.]), slots=2)
        self.assertEqual(cutoff, 5.)
        scores = np.array([7., 9., 8., 5.])
        report, admitted = replay_queue(events, scores, cutoff, capacity=1)
        _, flipped_admitted = replay_queue(events.assign(TX_FRAUD=1 - events["TX_FRAUD"]),
                                           scores, cutoff, capacity=1)
        self.assertEqual(admitted, flipped_admitted)
        self.assertEqual(admitted, {1, 4})
        self.assertEqual(report["overflow"], 2)
        self.assertEqual(report["caught"], 1)

    def test_distribution_shift_preserves_out_of_range_values(self):
        reference = np.arange(20)
        self.assertEqual(distribution_shift(reference, reference), 0)
        self.assertGreater(distribution_shift(reference, reference + 100), .5)

    def test_kaggle_report_reconciles_catches_and_queue_counts(self):
        report = json.loads((ROOT / "kaggle-report.json").read_text(encoding="utf-8"))
        facts = report["results"]["holdout"]
        for result in report["arrival_replay"].values():
            self.assertEqual(result["reviewed"] + result["overflow"], result["above_cutoff"])
            self.assertEqual(result["caught"] + result["missed"], facts["fraud"])
            self.assertEqual(result["caught"] + result["false_alerts"], result["reviewed"])
            self.assertTrue(all(b["reviewed"] <= report["fixed_budget"] for b in result["batches"]))


if __name__ == "__main__":
    unittest.main()
