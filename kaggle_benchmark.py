"""Evaluate fraud ranking on the real ULB / Worldline credit-card benchmark."""

from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment import DATA_DIR, FIXED_BUDGET, ROOT, SEED, rank_and_measure
from operations import distribution_shift, fit_cutoff, replay_queue

URL = "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"
PAGE = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
ARCHIVE_SHA256 = "a0360ce715992212e9ac72d8ccdca97f4be87dc1fdf2bed011358f7ab409a28a"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount"]
METHODS = {"amount": "Số tiền", "isolation_forest": "Isolation Forest",
           "logistic": "Logistic regression"}


def verify_archive(path):
    digest = sha256()
    with path.open("rb") as archive:
        while chunk := archive.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Kaggle ZIP checksum mismatch. Restore the pinned archive before proceeding.")


def read_source():
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "ulb-kaggle.zip"
    if not path.exists():
        with NamedTemporaryFile(dir=DATA_DIR, delete=False) as output:
            partial = Path(output.name)
            try:
                with urlopen(URL, timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                output.close()
                verify_archive(partial)
                partial.replace(path)
            finally:
                output.close()
                partial.unlink(missing_ok=True)
    verify_archive(path)
    with ZipFile(path) as archive, archive.open("creditcard.csv") as source:
        frame = pd.read_csv(source)
    expected = {"Time", "Amount", "Class", *FEATURES[:-1]}
    if len(frame) != 284807 or frame["Class"].sum() != 492 or set(frame) != expected:
        raise ValueError("Unexpected Kaggle dataset")
    return frame


def run():
    raw = read_source()
    raw["TRANSACTION_ID"] = raw.index
    raw["TX_FRAUD"] = raw["Class"]
    raw["log_amount"] = np.log1p(raw["Amount"])
    raw["day"] = (raw["Time"] // (6 * 3600)).astype(int)
    # Preserve the rank helper's group interface. Here one group is a six-hour batch.
    train, dev, holdout = (raw[raw["day"] < 4], raw[raw["day"].between(4, 5)],
                           raw[raw["day"] >= 6])
    print(f"Kaggle rows: {len(raw):,} · training: {len(train):,} · holdout: {len(holdout):,}", flush=True)
    isolation = IsolationForest(n_estimators=100, max_samples=256, random_state=SEED, n_jobs=2)
    isolation.fit(train[FEATURES])
    classifier = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    classifier.fit(train[FEATURES], train["Class"])
    results, thresholds = {}, {}
    for split, frame in (("dev", dev), ("holdout", holdout)):
        vector = frame[FEATURES]
        scorers = {"amount": frame["Amount"].to_numpy(), "isolation_forest": -isolation.score_samples(vector),
                   "logistic": classifier.decision_function(vector)}
        ranks, methods = {}, {}
        for name, values in scorers.items():
            ranks[name], methods[name] = rank_and_measure(frame, values)
        results[split] = {"rows": len(frame), "fraud": int(frame["Class"].sum()), "methods": methods}
        if split == "dev":
            winner = max(METHODS, key=lambda m: results[split]["methods"][m]["budgets"][FIXED_BUDGET]["tp"])
            slots = FIXED_BUDGET * frame["day"].nunique()
            thresholds = {name: fit_cutoff(values, slots) for name, values in scorers.items()}
            dev_scorers = scorers
    replays, selected = {}, {}
    diagnostics = []
    for name, values in scorers.items():
        replays[name], selected[name] = replay_queue(holdout, values, thresholds[name], FIXED_BUDGET)
    for batch in sorted(holdout["day"].unique()):
        mask = holdout["day"] == batch
        feature_shifts = {column: distribution_shift(dev[column].to_numpy(), holdout.loc[mask, column].to_numpy())
                          for column in FEATURES}
        diagnostics.append({"hour_range": [int(batch) * 6, (int(batch) + 1) * 6],
                            "logistic_score_js": distribution_shift(dev_scorers["logistic"], scorers["logistic"][mask]),
                            "isolation_score_js": distribution_shift(dev_scorers["isolation_forest"],
                                                                    scorers["isolation_forest"][mask]),
                            "largest_feature_shifts": dict(sorted(feature_shifts.items(), key=lambda x: -x[1])[:3])})
    chosen = ranks[winner]
    gaps = chosen[(chosen["rank"] <= FIXED_BUDGET) & (chosen["TX_FRAUD"] == 1) &
                  ~chosen["TRANSACTION_ID"].isin(selected[winner])]
    example = None
    if not gaps.empty:
        transaction_id = int(gaps.iloc[0]["TRANSACTION_ID"])
        event = holdout.loc[transaction_id]
        example = {"row_id": transaction_id, "batch": int(event["day"]), "label": 1, "methods": {}}
        for name, table in ranks.items():
            row = table.loc[transaction_id]
            example["methods"][name] = {"score": float(row["score"]), "rank": int(row["rank"]),
                                        "admitted_online": transaction_id in selected[name]}
    return {"source_page": PAGE, "download_url": URL, "archive_sha256": ARCHIVE_SHA256,
            "rows": len(raw), "fraud": int(raw["Class"].sum()), "train_rows": len(train),
            "train_fraud": int(train["Class"].sum()), "features": FEATURES, "fixed_budget": FIXED_BUDGET,
            "batch_hours": 6, "train_interval_hours": [0, 24], "dev_interval_hours": [24, 36],
            "holdout_interval_hours": [36, 48], "dev_winner": winner, "results": results,
            "arrival_replay": replays, "window_diagnostics": diagnostics, "queue_gap_case": example}


if __name__ == "__main__":
    report = run()
    target = ROOT / Path("kaggle-report.json")
    target.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    for method, score in report["results"]["holdout"]["methods"].items():
        print(method, score["ap"], score["budgets"][FIXED_BUDGET]["tp"], flush=True)
