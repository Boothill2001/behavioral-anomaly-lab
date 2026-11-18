import json
import unittest

import numpy as np
import pandas as pd

from experiment import ROOT
from queue_experiment import calibrate, main_choice_key, outcomes, route_events, valid_features
from kaggle_benchmark import FEATURES


class QueueTest(unittest.TestCase):
    def events(self, count):
        return pd.DataFrame({"TRANSACTION_ID": np.arange(count), "Time": np.arange(count),
                             "day": np.zeros(count, dtype=int), "TX_FRAUD": np.ones(count, dtype=int)})

    def test_reserved_quota_does_not_overflow_or_borrow(self):
        events = self.events(55)
        main = np.r_[np.ones(48), np.zeros(7)]
        ledger = route_events(events, main, np.ones(55), (.5, 1.), (45, 5))
        self.assertEqual((ledger["decision"] == "review").sum(), 50)
        self.assertEqual(ledger.loc[45, "decision"], "capacity_full")
        self.assertEqual(ledger.loc[45, "if_before"], 5)
        self.assertEqual(ledger.loc[48, "branch"], "if")
        self.assertEqual(ledger.loc[53, "decision"], "capacity_full")
        self.assertEqual(len(ledger.index.unique()), 55)

    def test_capacity_resets_and_labels_do_not_change_routing(self):
        events = self.events(52)
        events.loc[51, ["day", "Time"]] = [1, 21600]
        first = route_events(events, np.ones(52), np.zeros(52), (1., None), (50, 0))
        second = route_events(events.assign(TX_FRAUD=0), np.ones(52), np.zeros(52), (1., None), (50, 0))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.loc[50, "decision"], "capacity_full")
        self.assertEqual(first.loc[51, "main_before"], 50)
        self.assertEqual(first.loc[51, "decision"], "review")
        result = outcomes(events, first)
        self.assertEqual(result["caught"], 51)

    def test_invalid_inputs_do_not_consume_slots_and_ties_use_id(self):
        events = self.events(4).iloc[::-1].assign(Time=0)
        ledger = route_events(events, np.array([1., np.nan, 1., 1.]), np.ones(4),
                              (1., .5), (45, 5), valid=[True, True, False, True])
        self.assertEqual(list(ledger.index), [0, 1, 2, 3])
        self.assertEqual(ledger.loc[0, "decision"], "review")
        self.assertEqual(ledger.loc[1, "decision"], "invalid_data")
        self.assertEqual(ledger.loc[2, "decision"], "invalid_data")
        self.assertEqual(ledger.loc[3, "main_before"], 44)
        features = pd.DataFrame(np.ones((2, len(FEATURES))), columns=FEATURES)
        features.loc[1, "V1"] = np.inf
        self.assertEqual(valid_features(features).tolist(), [True, False])
        self.assertFalse(valid_features(features.drop(columns="V2")).any())
        features["V3"] = ["invalid", "1.0"]
        self.assertFalse(valid_features(features).any())
        duplicate = self.events(2).assign(TRANSACTION_ID=0)
        with self.assertRaises(ValueError):
            route_events(duplicate, np.ones(2), np.ones(2), (1., .5), (45, 5))

    def test_no_eligible_if_candidates_disables_secondary_cutoff(self):
        values = np.ones(120)
        cutoff = calibrate(values, values, (45, 5))
        self.assertEqual(cutoff, (1., None))

    def test_report_selection_and_all_policy_counts_reconcile(self):
        path = ROOT / "xgb-queue-report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        selection = report["selection"]
        saved_selection = json.loads((ROOT / "xgb-queue-selection.json").read_text(encoding="utf-8"))
        self.assertEqual(selection, saved_selection)
        self.assertNotIn("Class", report["features"])
        self.assertNotIn("TX_FRAUD", report["features"])
        winner = min(selection["candidates"], key=main_choice_key)
        self.assertEqual(winner["name"], selection["main"])
        gate = all(
            selection["dev_policies"][f"reserved_45_5_{seed}"]["caught"] >
            selection["dev_policies"][f"main_50_{seed}"]["caught"] and
            selection["dev_policies"][f"reserved_45_5_{seed}"]["false_alerts"] <=
            selection["dev_policies"][f"main_50_{seed}"]["false_alerts"]
            for seed in report["seed_values"]
        )
        self.assertEqual(selection["policy"], "reserved_45_5" if gate else "main_50")
        for key, result in report["results"].items():
            value = result["arrival"]
            self.assertEqual(value["caught"] + value["missed"], 115)
            self.assertEqual(value["caught"] + value["false_alerts"], value["reviewed"])
            self.assertEqual(sum(b["reviewed"] for b in value["branches"].values()), value["reviewed"])
            self.assertTrue(all(batch["reviewed"] <= 50 for batch in value["batches"]))
            if key.startswith("reserved"):
                self.assertTrue(all(batch["main_remaining"] <= 45 and batch["if_remaining"] <= 5
                                    for batch in value["batches"]))
        for seed in report["seed_values"]:
            delta = report["paired"][str(seed)]
            a = report["results"][f"main_50_{seed}"]["arrival"]["caught"]
            b = report["results"][f"reserved_45_5_{seed}"]["arrival"]["caught"]
            self.assertEqual(b - a, delta["rescued"] - delta["regressed"])


if __name__ == "__main__":
    unittest.main()
