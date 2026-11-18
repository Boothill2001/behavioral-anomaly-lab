"""Real model replay with explicitly simulated review sampling and label delays."""
from hashlib import sha256
import gzip
from io import StringIO
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import xgboost

from experiment import ROOT
from if_research import classifier, forest, save_json
from kaggle_benchmark import ARCHIVE_SHA256, FEATURES, read_source
from operations import fit_cutoff, replay_queue
from queue_experiment import make_xgb

QA_SEED = 20261118
QA_FRACTION = .1
MODELS = ("xgb_champion", "logistic_challenger", "if5_challenger")
IF_FEATURES = ["V17", "V14", "V12", "V10", "V11"]


def sample_reviews(event_ids, champion_admitted, fraction=QA_FRACTION, seed=QA_SEED):
    """Census of admitted cases + simple random sampling without replacement elsewhere."""
    ids = np.asarray(event_ids, dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or not 0 < fraction <= 1:
        raise ValueError("Unique event IDs and positive sampling fraction required")
    champion = np.asarray(champion_admitted, dtype=bool)
    if champion.shape != ids.shape:
        raise ValueError("Decision mask must align with event IDs")
    others = np.flatnonzero(~champion)
    n = min(len(others), max(1, int(np.ceil(len(others) * fraction)))) if len(others) else 0
    rng = np.random.default_rng(seed)
    sampled = champion.copy()
    if n:
        sampled[rng.choice(others, size=n, replace=False)] = True
    probability = np.ones(len(ids), dtype=float)
    if n:
        probability[others] = n / len(others)
    # Delays are generated independently of labels and predictions, after cohort closure.
    delay_days = rng.integers(1, 8, size=len(ids))
    return sampled, probability, delay_days


def joined_labels(ledger, as_of_day):
    """Missing/pending human verification is unknown, never a negative label."""
    known = ledger["sampled"] & (ledger["verified_day"] <= as_of_day)
    labels = ledger["human_label"].where(known, np.nan)
    return labels


def weighted_metrics(labels, decisions, probabilities):
    """Horvitz–Thompson totals and ratio estimates on a fully matured review sample."""
    y = np.asarray(labels, dtype=float)
    d = np.asarray(decisions, dtype=bool)
    pi = np.asarray(probabilities, dtype=float)
    if not (y.shape == d.shape == pi.shape) or np.any((pi <= 0) | (pi > 1)):
        raise ValueError("Aligned data with inclusion probabilities in (0,1] required")
    observed = np.isfinite(y)
    if not np.isin(y[observed], [0, 1]).all():
        raise ValueError("Verified labels must be binary")
    w = 1 / pi[observed]
    yy, dd = y[observed], d[observed]
    tp = float(np.sum(w * yy * dd))
    fn = float(np.sum(w * yy * ~dd))
    fp = float(np.sum(w * (1 - yy) * dd))
    return {"reviewed_labels": int(observed.sum()), "estimated_tp": tp, "estimated_fn": fn,
            "estimated_fp": fp, "precision_estimate": tp / (tp + fp) if tp + fp else None,
            "recall_estimate": tp / (tp + fn) if tp + fn else None}


def population_stability(reference, observed):
    """PSI on development quantile bins, with Jeffreys pseudocount smoothing."""
    reference, observed = np.asarray(reference), np.asarray(observed)
    if not len(reference) or not len(observed) or not np.isfinite(reference).all() or not np.isfinite(observed).all():
        raise ValueError("PSI requires nonempty finite score arrays")
    edges = np.r_[-np.inf, np.unique(np.quantile(reference, np.arange(.1, 1, .1))), np.inf]
    a = np.histogram(reference, edges)[0] + .5
    b = np.histogram(observed, edges)[0] + .5
    p, q = a / a.sum(), b / b.sum()
    return {"psi": float(np.sum((q - p) * np.log(q / p))), "bin_count": len(p),
            "inner_edges": list(map(float, edges[1:-1])), "reference_proportions": p.tolist(),
            "observed_proportions": q.tolist(), "smoothing": "0.5 pseudocount per bin, then renormalize"}


def timing(score, batch):
    score(batch)
    elapsed = []
    for _ in range(50):
        before = perf_counter()
        score(batch)
        elapsed.append((perf_counter() - before) * 1000)
    return {"p95_ms": float(np.quantile(elapsed, .95)), "p50_ms": float(np.median(elapsed)),
            "batch_rows": len(batch), "repeats": len(elapsed), "device": "CPU",
            "scope": "scoring a 1024-row batch after one warmup; not per-event or end-to-end latency"}


def stage(ledger, day):
    y = joined_labels(ledger, day)
    ready = bool(y.notna().sum() == ledger["sampled"].sum())
    outcomes = {}
    for m in MODELS:
        # Incomplete maturation may induce additional sampling bias; block inference.
        outcomes[m] = weighted_metrics(y, ledger[m + "_admitted"], ledger["inclusion_probability"]) if ready else None
    return {"as_of_day_after_cohort_close": day, "verified": int(y.notna().sum()),
            "pending_reviews": int(ledger["sampled"].sum() - y.notna().sum()),
            "unknown_total": int(y.isna().sum()), "label_coverage": float(y.notna().mean()),
            "sample_mature": ready, "metrics": outcomes,
            "metric_status": "sample-weighted estimates; not census recall" if ready else "blocked: review sample not fully matured"}


def run():
    data = read_source()
    data["TRANSACTION_ID"] = data.index
    data["TX_FRAUD"] = data["Class"]
    data["log_amount"] = np.log1p(data["Amount"])
    data["day"] = (data["Time"] // 21600).astype(int)
    train, dev, hold = data[data["day"] < 4], data[data["day"].between(4, 5)], data[data["day"] >= 6]
    if not np.isfinite(data[FEATURES]).all().all():
        raise ValueError("Invalid feature inputs")
    fitted = {"xgb_champion": make_xgb(4, False, 1, 42).fit(train[FEATURES], train["Class"]),
              "logistic_challenger": classifier().fit(train[FEATURES], train["Class"]),
              "if5_challenger": forest((1024, 1.), 42).fit(train.loc[train["Class"] == 0, IF_FEATURES])}
    ledger = hold[["TRANSACTION_ID", "Time"]].rename(columns={"TRANSACTION_ID": "event_id"}).copy()
    model_reports = {}
    for name, model in fitted.items():
        columns = IF_FEATURES if name == "if5_challenger" else FEATURES
        def score(frame):
            return -model.score_samples(frame[columns]) if name == "if5_challenger" else model.predict_proba(frame[columns])[:, 1]
        development = score(dev)
        values = score(hold)
        repeated = score(hold)
        cutoff = fit_cutoff(development, 100)
        _, admitted = replay_queue(hold, values, cutoff, 50)
        ledger[name + "_score"] = values
        ledger[name + "_admitted"] = ledger["event_id"].isin(admitted)
        version = sha256(pickle.dumps(model, protocol=5)).hexdigest()
        model_reports[name] = {"model_state_sha256": version, "features": columns, "cutoff": cutoff,
                               "role": "champion" if name == "xgb_champion" else "shadow only",
                               "training_rows": len(train) if name != "if5_challenger" else int((train["Class"] == 0).sum()),
                               "training_end_seconds": float(train["Time"].max()),
                               "reference_rows": len(dev), "scored_rows": len(hold), "error_count": 0,
                               "repeat_max_abs_score_delta": float(np.max(np.abs(values - repeated))),
                               "repeat_decision_disagreements": int(((values >= cutoff) != (repeated >= cutoff)).sum()),
                               "score_shift": population_stability(development, values),
                               "timing": timing(score, hold.iloc[:1024])}
        print(name, "admitted", len(admitted), "PSI", model_reports[name]["score_shift"]["psi"], flush=True)
    sampled, probabilities, delays = sample_reviews(ledger["event_id"], ledger["xgb_champion_admitted"])
    ledger["sampled"] = sampled
    ledger["inclusion_probability"] = probabilities
    ledger["verified_day"] = np.where(sampled, delays, np.nan)
    ledger["human_label"] = np.where(sampled, hold["Class"].to_numpy(), np.nan)
    ledger["feature_sha256"] = [sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()
                                 for values in hold[FEATURES].to_numpy()]
    events_hash = sha256(ledger["event_id"].to_numpy(dtype="<i8").tobytes()).hexdigest()
    for m in MODELS:
        model_reports[m]["event_ids_sha256"] = events_hash
    final = stage(ledger, 7)
    oracle = {}
    for m in MODELS:
        # Offline audit only. Never feed this census back into sampling or fit/threshold selection.
        oracle[m] = weighted_metrics(hold["Class"], ledger[m + "_admitted"], np.ones(len(hold)))
    cases = []
    all_normal = ~ledger[[m + "_admitted" for m in MODELS]].any(axis=1)
    selections = {
        "all_models_skip_confirmed_fraud": ledger[all_normal & (hold["Class"] == 1)],
        "reviewed_champion_challenger_disagreement": ledger[ledger["sampled"] & (ledger["xgb_champion_admitted"] != ledger["logistic_challenger_admitted"])],
        "qa_found_champion_miss": ledger[ledger["sampled"] & ~ledger["xgb_champion_admitted"] & (hold["Class"] == 1)],
    }
    for group, subset in selections.items():
        for idx, row in subset.head(10).iterrows():
            entry = {"group": group, "event_id": int(row.event_id), "sampled": bool(row.sampled),
                     "verified_day": int(row.verified_day) if row.sampled else None,
                     "human_label": int(row.human_label) if row.sampled else None,
                     "offline_census_audit_label": int(hold.loc[idx, "Class"]),
                     "input_features": {c: float(hold.loc[idx, c]) for c in FEATURES},
                     "inclusion_probability": float(row.inclusion_probability),
                     "models": {m: {"score": float(row[m + "_score"]), "admitted": bool(row[m + "_admitted"])} for m in MODELS}}
            cases.append(entry)
    content = ledger.to_csv(index=False, float_format="%.17g").encode()
    compressed = gzip.compress(content, mtime=0)
    (ROOT / "shadow-ledger.csv.gz").write_bytes(compressed)
    n_non = int((~ledger["xgb_champion_admitted"]).sum())
    qa_labels = ledger["sampled"] & ~ledger["xgb_champion_admitted"]
    paired = {m: {"decision_disagreements_with_champion": int((ledger[m + "_admitted"] != ledger["xgb_champion_admitted"]).sum()),
                  "both_skip": int((~ledger[m + "_admitted"] & ~ledger["xgb_champion_admitted"]).sum())} for m in MODELS[1:]}
    output = {"scope": "Retrospective replay on real ULB transactions; review sampling and 1–7 day label delays are simulated.",
              "source": {"page": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud", "archive_sha256": ARCHIVE_SHA256,
                         "dataset_rows": len(data), "cohort_rows": len(hold), "cohort_relative_hours": [36, 48],
                         "warning": "Two days of source traffic, not one week of production traffic; evaluation window has been inspected in prior research."},
              "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
              "source_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "models": model_reports,
              "sampling": {"seed": QA_SEED, "design": "100% champion queue admissions; SRS without replacement from all others",
                           "champion_census": int(ledger["xgb_champion_admitted"].sum()), "nonadmitted_population": n_non,
                           "nonadmitted_sample": int(qa_labels.sum()), "nonadmitted_inclusion_probability": float(qa_labels.sum() / n_non),
                           "qa_positive_labels": int(ledger.loc[qa_labels, "human_label"].sum()),
                           "warning": "QA reviews are an additional experimental workload, outside the 50-slot production queue. Estimates assume completed, correct review labels and the recorded sampling design."},
              "snapshots": [stage(ledger, d) for d in (0, 2, 7)], "paired": paired, "cases": cases,
              "offline_census_audit_only": oracle,
              "ledger": {"file": "shadow-ledger.csv.gz", "sha256": sha256(compressed).hexdigest(), "rows": len(ledger), "event_ids_sha256": events_hash},
              "release_decision": {"decision": "HOLD", "reasons": ["No fresh, previously unseen release cohort", "Synthetic label delays do not validate a real human review process", "Sparse QA positives yield uncertain missed-fraud estimates; no confidence-bound release gate is established", "No real end-to-end serving parity or production SLO has been measured"],
                                   "note": "Score PSI is a distribution diagnostic, not proof of concept drift. Repeat prediction parity is not accuracy. No universal 10% tolerance is assumed."}}
    save_json(ROOT / "shadow-report.json", output)
    render(output)
    return output


def render(report):
    payload = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    template = (ROOT / "shadow-template.html").read_text(encoding="utf-8")
    (ROOT / "shadow.html").write_text(template.replace("__SHADOW__", payload), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render(json.loads((ROOT / "shadow-report.json").read_text(encoding="utf-8")))
    else:
        run()
