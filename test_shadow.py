"""Selection bias, delayed labels and parity are distinct from predictive correctness."""
from hashlib import sha256
import gzip
from io import StringIO
import json
import unittest

import numpy as np
import pandas as pd
import shadow_evaluation as s


class ShadowTests(unittest.TestCase):
    def test_identical_normal_predictions_can_both_miss_fraud(self):
        labels = [1, 0]
        first = second = [False, False]
        self.assertEqual(first, second)
        metrics = s.weighted_metrics(labels, first, [1., 1.])
        self.assertEqual(metrics["recall_estimate"], 0.)
        self.assertEqual(metrics["estimated_fn"], 1.)

    def test_pending_and_unsampled_labels_never_become_normal(self):
        ledger = pd.DataFrame({"sampled": [True, True, False], "verified_day": [1, 7, np.nan],
                               "human_label": [0., 1., np.nan]})
        early = s.joined_labels(ledger, 2)
        self.assertEqual(early.iloc[0], 0.)
        self.assertTrue(early.iloc[1:].isna().all())
        late = s.joined_labels(ledger, 7)
        self.assertEqual(late.iloc[1], 1.)
        self.assertTrue(pd.isna(late.iloc[2]))

    def test_sampling_covers_champion_and_is_label_blind(self):
        ids = np.arange(101)
        admitted = ids < 1
        selected, probability, delays = s.sample_reviews(ids, admitted, .1, 42)
        self.assertTrue(selected[0])
        self.assertEqual(selected[1:].sum(), 10)
        np.testing.assert_array_equal(probability[1:], .1)
        self.assertEqual(probability[0], 1)
        self.assertTrue(((delays >= 1) & (delays <= 7)).all())
        again = s.sample_reviews(ids, admitted, .1, 42)
        np.testing.assert_array_equal(selected, again[0])
        with self.assertRaises(ValueError):
            s.sample_reviews([1, 1], [True, False])

    def test_inverse_probability_weights_correct_review_selection_bias(self):
        # Two alerted frauds are a census. Two reviewed non-alert frauds represent twenty.
        labels = [1, 1, 1, 1]
        admitted = [True, True, False, False]
        weighted = s.weighted_metrics(labels, admitted, [1., 1., .1, .1])
        naive = s.weighted_metrics(labels, admitted, [1.] * 4)
        self.assertAlmostEqual(weighted["recall_estimate"], 2 / 22)
        self.assertEqual(naive["recall_estimate"], .5)
        self.assertEqual(weighted["estimated_fn"], 20)
        with self.assertRaises(ValueError):
            s.weighted_metrics([1], [True], [0])

    def test_psi_is_zero_for_same_distribution_and_retains_extremes(self):
        reference = np.arange(100)
        self.assertEqual(s.population_stability(reference, reference)["psi"], 0.)
        shift = s.population_stability(reference, np.full(100, 999))
        self.assertGreater(shift["psi"], 0)
        self.assertAlmostEqual(sum(shift["observed_proportions"]), 1.)

    def test_committed_ledger_and_reports_reconcile(self):
        report = json.loads((s.ROOT / "shadow-report.json").read_text(encoding="utf-8"))
        zipped = (s.ROOT / report["ledger"]["file"]).read_bytes()
        self.assertEqual(sha256(zipped).hexdigest(), report["ledger"]["sha256"])
        ledger = pd.read_csv(StringIO(gzip.decompress(zipped).decode()))
        self.assertEqual(len(ledger), report["ledger"]["rows"])
        self.assertTrue(ledger["event_id"].is_unique)
        ids_hash = sha256(ledger["event_id"].to_numpy(dtype="<i8").tobytes()).hexdigest()
        self.assertEqual(ids_hash, report["ledger"]["event_ids_sha256"])
        self.assertTrue(ledger.loc[~ledger["sampled"], "human_label"].isna().all())
        final = s.stage(ledger, 7)
        for model in s.MODELS:
            self.assertEqual(report["models"][model]["event_ids_sha256"], ids_hash)
            for key, value in final["metrics"][model].items():
                self.assertAlmostEqual(value, report["snapshots"][-1]["metrics"][model][key])
        for day in (0, 2):
            self.assertFalse(s.stage(ledger, day)["sample_mature"])
            self.assertTrue(all(v is None for v in s.stage(ledger, day)["metrics"].values()))
        self.assertEqual(report["release_decision"]["decision"], "HOLD")


if __name__ == "__main__":
    unittest.main()
