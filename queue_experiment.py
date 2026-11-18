"""Compare supervised scorers and a reserved 45/5 arrival review queue."""

from hashlib import sha256
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBClassifier

from experiment import ROOT, rank_and_measure
from if_research import classifier, forest, paired_changes, save_json
from kaggle_benchmark import ARCHIVE_SHA256, FEATURES, read_source
from operations import fit_cutoff

SEEDS = (42, 43, 44)
GRID = [(depth, weighted) for depth in (2, 4, 6) for weighted in (False, True)]


def valid_features(frame):
    numeric = frame.reindex(columns=FEATURES).apply(pd.to_numeric, errors="coerce")
    return np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)


def route_events(events, main_scores, if_scores, cutoffs, capacity, valid=None):
    """Route in arrival order, using scores and quota state only."""
    if len(events) != len(main_scores) or len(events) != len(if_scores):
        raise ValueError("Score arrays must align with events")
    if any(c < 0 for c in capacity) or sum(capacity) != 50:
        raise ValueError("The two reserved capacities must total 50")
    if not events["TRANSACTION_ID"].is_unique:
        raise ValueError("Duplicate transaction IDs")
    valid = np.ones(len(events), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != (len(events),):
        raise ValueError("Validity mask must align with events")
    inputs = events[["Time", "day", "TRANSACTION_ID"]].assign(
        main_score=main_scores, if_score=if_scores, data_valid=valid)
    inputs = inputs.sort_values(["Time", "TRANSACTION_ID"])
    batch = None
    remaining = list(capacity)
    records = []
    for event in inputs.itertuples(index=False):
        if event.day != batch:
            remaining = list(capacity)
            batch = event.day
        before = remaining.copy()
        score_valid = np.isfinite(event.main_score) and np.isfinite(event.if_score)
        branch, decision = "none", "below_cutoffs"
        if not event.data_valid or not score_valid:
            decision = "invalid_data"
        elif cutoffs[0] is not None and event.main_score >= cutoffs[0]:
            branch = "main"
            decision = "review" if remaining[0] else "capacity_full"
            if remaining[0]:
                remaining[0] -= 1
        elif capacity[1] and cutoffs[1] is not None and event.if_score >= cutoffs[1]:
            branch = "if"
            decision = "review" if remaining[1] else "capacity_full"
            if remaining[1]:
                remaining[1] -= 1
        records.append({
            "row_id": int(event.TRANSACTION_ID), "batch": int(batch),
            "main_score": float(event.main_score), "if_score": float(event.if_score),
            "branch": branch, "decision": decision, "main_before": before[0], "if_before": before[1],
            "main_after": remaining[0], "if_after": remaining[1],
        })
    return pd.DataFrame.from_records(records).set_index("row_id")


def calibrate(dev_main, dev_if, split):
    main_slots = split[0] * 2
    main_cutoff = fit_cutoff(dev_main, main_slots)
    candidates = dev_if[dev_main < main_cutoff]
    if_cutoff = None
    if split[1] and len(candidates):
        if_cutoff = fit_cutoff(candidates, min(split[1] * 2, len(candidates)))
    return (main_cutoff, if_cutoff)


def outcomes(events, ledger):
    labels = events.set_index("TRANSACTION_ID")["TX_FRAUD"]
    reviewed = ledger[ledger["decision"] == "review"]
    caught = int(labels.reindex(reviewed.index).sum())
    fraud = int(labels.sum())
    n = len(reviewed)
    branches = {}
    for name in ("main", "if"):
        branch = reviewed[reviewed["branch"] == name]
        tp = int(labels.reindex(branch.index).sum())
        branches[name] = {"reviewed": len(branch), "caught": tp, "false_alerts": len(branch) - tp}
    batches = []
    for batch, rows in ledger.groupby("batch", sort=True):
        seen = rows[rows["decision"] == "review"]
        tp = int(labels.reindex(seen.index).sum())
        batches.append({"hours": [int(batch) * 6, (int(batch) + 1) * 6], "rows": len(rows),
                        "reviewed": len(seen), "caught": tp, "false_alerts": len(seen) - tp,
                        "fraud": int(labels.reindex(rows.index).sum()),
                        "overflow": int((rows["decision"] == "capacity_full").sum()),
                        "main_remaining": int(rows.iloc[-1]["main_after"]),
                        "if_remaining": int(rows.iloc[-1]["if_after"])})
    return {"caught": caught, "false_alerts": n - caught, "missed": fraud - caught,
            "reviewed": n, "fraud": fraud, "precision": caught / n if n else 0,
            "recall": caught / fraud if fraud else 0,
            "invalid": int((ledger["decision"] == "invalid_data").sum()),
            "overflow": int((ledger["decision"] == "capacity_full").sum()),
            "branches": branches, "batches": batches}


def evaluate(events, main_scores, if_scores, cutoffs, split):
    ledger = route_events(events, main_scores, if_scores, cutoffs, split, valid_features(events))
    return outcomes(events, ledger), ledger


def reviewed_ids(ledger):
    return set(map(int, ledger.index[ledger["decision"] == "review"]))


def timing(model, batch):
    model.predict_proba(batch)
    elapsed = []
    for _ in range(10):
        start = perf_counter()
        model.predict_proba(batch)
        elapsed.append((perf_counter() - start) * 1000)
    return {"rows_per_batch": len(batch), "repeats": 10, "p50_ms": float(np.median(elapsed)),
            "p95_ms": float(np.quantile(elapsed, .95)), "device": "CPU",
            "includes": "predict_proba only, one warmup, fixed 1024-row evaluation batch"}


def make_xgb(depth, weighted, ratio, seed):
    return XGBClassifier(n_estimators=200, max_depth=depth, learning_rate=.05,
                         subsample=.8, colsample_bytree=.8, n_jobs=2, random_state=seed,
                         scale_pos_weight=ratio if weighted else 1, tree_method="hist",
                         device="cpu", objective="binary:logistic", eval_metric="logloss")


def main_choice_key(row):
    return (-row["mean_caught"], row["mean_false_alerts"], row["std_caught"],
            0 if row["method"] == "logistic" else 1, row.get("depth", 0), row.get("weighted", False))


def run():
    raw = read_source()
    raw["TRANSACTION_ID"] = raw.index
    raw["TX_FRAUD"] = raw["Class"]
    raw["log_amount"] = np.log1p(raw["Amount"])
    raw["day"] = (raw["Time"] // 21600).astype(int)
    train, dev, hold = raw[raw["day"] < 4], raw[raw["day"].between(4, 5)], raw[raw["day"] >= 6]
    if not valid_features(raw).all():
        raise ValueError("Source has invalid model inputs")
    ratio = float((train["Class"] == 0).sum() / train["Class"].sum())
    logistic = classifier().fit(train[FEATURES], train["Class"])
    forests = {seed: forest((1024, 1.0), seed).fit(train[FEATURES]) for seed in SEEDS}
    dev_if = {seed: -forests[seed].score_samples(dev[FEATURES]) for seed in SEEDS}
    candidates, fitted = [], {}
    settings = [("logistic", 0, False), *[(f"xgb_d{d}_{'weighted' if w else 'unweighted'}", d, w)
                                          for d, w in GRID]]
    for name, depth, weighted in settings:
        runs = []
        for seed in SEEDS:
            model = logistic if name == "logistic" else make_xgb(depth, weighted, ratio, seed).fit(
                train[FEATURES], train["Class"])
            scores = model.predict_proba(dev[FEATURES])[:, 1]
            cutoffs = calibrate(scores, dev_if[seed], (50, 0))
            outcome, _ = evaluate(dev, scores, dev_if[seed], cutoffs, (50, 0))
            fitted[(name, seed)] = (model, scores)
            runs.append({"seed": seed, "cutoffs": cutoffs, **outcome})
        caught = [row["caught"] for row in runs]
        row = {"name": name, "method": "logistic" if not depth else "xgboost",
               "depth": depth, "weighted": weighted, "scale_pos_weight": ratio if weighted else 1,
               "mean_caught": float(np.mean(caught)), "std_caught": float(np.std(caught)),
               "mean_false_alerts": float(np.mean([r["false_alerts"] for r in runs])), "runs": runs}
        candidates.append(row)
        print(name, "dev", caught, flush=True)
    main_winner = min(candidates, key=main_choice_key)["name"]
    dev_policy = {}
    calibrated = {}
    for seed in SEEDS:
        scores = fitted[(main_winner, seed)][1]
        for name, split in (("main_50", (50, 0)), ("reserved_45_5", (45, 5))):
            cutoff = calibrate(scores, dev_if[seed], split)
            outcome, _ = evaluate(dev, scores, dev_if[seed], cutoff, split)
            calibrated[(seed, name)] = cutoff
            dev_policy[f"{name}_{seed}"] = outcome
    mixed_passed = all(
        dev_policy[f"reserved_45_5_{seed}"]["caught"] > dev_policy[f"main_50_{seed}"]["caught"] and
        dev_policy[f"reserved_45_5_{seed}"]["false_alerts"] <= dev_policy[f"main_50_{seed}"]["false_alerts"]
        for seed in SEEDS)
    selection = {"main": main_winner, "policy": "reserved_45_5" if mixed_passed else "main_50",
                 "rule": "dev arrival catches desc, false alerts asc, seed variation asc, then simpler model",
                 "policy_rule": "45/5 must increase catches with no extra false alerts in every seed on dev",
                 "candidates": candidates, "dev_policies": dev_policy,
                 "policy_cutoffs": {f"{name}_{seed}": value for (seed, name), value in calibrated.items()}}
    save_json(ROOT / "xgb-queue-selection.json", selection)
    print("Frozen on dev:", main_winner, selection["policy"], flush=True)
    hold_if = {seed: -forests[seed].score_samples(hold[FEATURES]) for seed in SEEDS}
    results, ledgers, timings, paired = {}, {}, {}, {}
    for seed in SEEDS:
        main_model = fitted[(main_winner, seed)][0]
        main_scores = main_model.predict_proba(hold[FEATURES])[:, 1]
        if_scores = hold_if[seed]
        for name, split in (("main_50", (50, 0)), ("reserved_45_5", (45, 5))):
            outcome, ledger = evaluate(hold, main_scores, if_scores, calibrated[(seed, name)], split)
            _, batch = rank_and_measure(hold, main_scores)
            key = f"{name}_{seed}"
            results[key] = {"arrival": outcome, "main_score_batch_top_k": batch,
                            "cutoffs": calibrated[(seed, name)], "capacity": split}
            ledgers[key] = ledger
            print(key, outcome["caught"], "fraud", outcome["false_alerts"], "false alerts", flush=True)
        paired[str(seed)] = paired_changes(
            hold.set_index("TRANSACTION_ID")["TX_FRAUD"],
            reviewed_ids(ledgers[f"main_50_{seed}"]), reviewed_ids(ledgers[f"reserved_45_5_{seed}"]))
        timings[f"selected_main_{seed}"] = timing(main_model, hold[FEATURES].iloc[:1024])

    # Score each XGB candidate on the old evaluation window after freezing selection.
    model_results = {}
    for candidate in candidates:
        name = candidate["name"]
        for seed in SEEDS:
            model, dev_scores = fitted[(name, seed)]
            values = model.predict_proba(hold[FEATURES])[:, 1]
            if_scores = hold_if[seed]
            cutoff = calibrate(dev_scores, dev_if[seed], (50, 0))
            outcome, _ = evaluate(hold, values, if_scores, cutoff, (50, 0))
            _, batch = rank_and_measure(hold, values)
            model_results[f"{name}_{seed}"] = {"arrival": outcome, "batch_top_k": batch}
    timings["logistic"] = timing(logistic, hold[FEATURES].iloc[:1024])
    groups = {}
    labels = hold.set_index("TRANSACTION_ID")["TX_FRAUD"]
    fraud_ids = set(labels.index[labels == 1])
    for seed in SEEDS:
        groups[f"rescued_{seed}"] = paired[str(seed)]["rescued_ids"][:3]
        groups[f"regressed_{seed}"] = paired[str(seed)]["regressed_ids"][:3]
        mixed = ledgers[f"reserved_45_5_{seed}"]
        if_ids = set(mixed.index[(mixed["branch"] == "if") & (mixed["decision"] == "review")])
        groups[f"if_false_alerts_{seed}"] = sorted(map(int, if_ids - fraud_ids))[:3]
    groups["missed_both"] = sorted(map(int, fraud_ids - reviewed_ids(ledgers["main_50_42"])
                                      - reviewed_ids(ledgers["reserved_45_5_42"])))[:3]
    groups["caught_main"] = sorted(map(int, fraud_ids & reviewed_ids(ledgers["main_50_42"])))[:3]
    ids = sorted({row_id for group in groups.values() for row_id in group})
    cases = []
    for row_id in ids:
        event = hold.loc[row_id]
        traces = {key: ledger.loc[row_id].to_dict() for key, ledger in ledgers.items()}
        cases.append({"row_id": row_id, "label": int(event["TX_FRAUD"]), "time_seconds": float(event["Time"]),
                      "amount": float(event["Amount"]),
                      "features": {feature: float(event[feature]) for feature in FEATURES}, "traces": traces})
    return {"status": "complete", "scope": "Exploratory research on previously inspected Kaggle ULB/Worldline data",
            "archive_sha256": ARCHIVE_SHA256, "source_code_sha256": sha256((ROOT / "queue_experiment.py").read_bytes()).hexdigest(),
            "versions": {"xgboost": xgboost.__version__, "sklearn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__},
            "features": FEATURES, "seed_values": SEEDS, "train_rows": len(train),
            "dev_rows": len(dev), "evaluation_rows": len(hold), "evaluation_fraud": int(hold["Class"].sum()),
            "selection": selection, "results": results, "models": model_results, "timings": timings,
            "paired": paired, "case_groups": groups, "cases": cases}


if __name__ == "__main__":
    save_json(ROOT / "xgb-queue-report.json", run())
