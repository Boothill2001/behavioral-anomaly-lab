"""Investigate IF misses and compare a dev-selected forest and an OOF hybrid."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment import FIXED_BUDGET, ROOT, rank_and_measure
from kaggle_benchmark import ARCHIVE_SHA256, FEATURES, read_source
from operations import distribution_shift, fit_cutoff, replay_queue

SEEDS = (42, 43, 44)
GRID = [(samples, features) for samples in (256, 1024, 4096) for features in (.5, 1.0)]


def save_json(path, value):
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as output:
        partial = Path(output.name)
        try:
            json.dump(value, output, allow_nan=False, ensure_ascii=False, indent=2)
            output.write("\n")
        except BaseException:
            output.close()
            partial.unlink(missing_ok=True)
            raise
    try:
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def forest(config, seed):
    samples, features = config
    return IsolationForest(n_estimators=100, max_samples=samples, max_features=features,
                           random_state=seed, n_jobs=2)


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))


def config_name(config):
    return f"samples_{config[0]}_features_{config[1]:g}"


def evaluate(frame, scores, cutoff):
    ranking, metrics = rank_and_measure(frame, scores)
    replay, admitted = replay_queue(frame, scores, cutoff, FIXED_BUDGET)
    by_id = ranking.set_index("TRANSACTION_ID")
    return {"metrics": metrics, "arrival": replay}, by_id, admitted


def crossfit_scores(train, config, seed, *, factory=forest):
    """Each training row is scored by a forest that never fitted that row."""
    out_of_fold = np.full(len(train), np.nan)
    fold_ids = np.full(len(train), -1, dtype=np.int8)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (fitted, scored) in enumerate(splitter.split(train)):
        model = factory(config, seed)
        model.fit(train.iloc[fitted])
        out_of_fold[scored] = -model.score_samples(train.iloc[scored])
        fold_ids[scored] = fold
    if not np.isfinite(out_of_fold).all() or (fold_ids < 0).any():
        raise ValueError("Incomplete out-of-fold coverage")
    return out_of_fold, fold_ids


def augmented(frame, scores):
    return frame.assign(if_anomaly_score=scores)


def paired_changes(labels, before, after):
    fraud = set(labels.index[labels == 1])
    return {"rescued": len(fraud & (after - before)), "regressed": len(fraud & (before - after)),
            "rescued_ids": sorted(map(int, fraud & (after - before))),
            "regressed_ids": sorted(map(int, fraud & (before - after)))}


def summarize_windows(frame, dev, dev_scores, scores, ranking, admitted):
    windows = []
    for batch, events in frame.groupby("day", sort=True):
        idx = frame.index.get_indexer(events.index)
        score = scores[idx]
        labels = events["TX_FRAUD"].to_numpy()
        window = {"hours": [int(batch) * 6, (int(batch) + 1) * 6],
                  "rows": len(events), "fraud": int(labels.sum()),
                  "score_js_vs_dev": distribution_shift(dev_scores, score),
                  "score_by_label": {}}
        for label in (0, 1):
            values = score[labels == label]
            window["score_by_label"][str(label)] = {
                "n": len(values),
                "p50": float(np.median(values)) if len(values) else None,
                "p95": float(np.quantile(values, .95)) if len(values) else None,
            }
        fraud_ids = events.loc[events["TX_FRAUD"] == 1, "TRANSACTION_ID"]
        fraud_ranks = ranking.loc[fraud_ids, "rank"]
        window["fraud_top_k"] = {str(k): int((fraud_ranks <= k).sum()) for k in (50, 100, 200)}
        window["admitted"] = sum(int(i) in admitted for i in events["TRANSACTION_ID"])
        shifts = {column: distribution_shift(dev[column].to_numpy(), events[column].to_numpy())
                  for column in FEATURES}
        window["largest_feature_shifts"] = dict(sorted(shifts.items(), key=lambda item: (-item[1], item[0]))[:5])
        windows.append(window)
    return windows


def case_rows(frame, train, rankings, admissions, summaries, groups):
    center = train[FEATURES].mean()
    spread = train[FEATURES].std().replace(0, 1)
    records = []
    membership = {}
    for group, ids in groups.items():
        for row_id in ids:
            membership.setdefault(int(row_id), []).append(group)
    for row_id, categories in membership.items():
        row = frame.loc[row_id]
        deviations = ((row[FEATURES].astype(float) - center) / spread).abs()
        top_features = deviations.sort_values(ascending=False).head(5).index
        observations = {}
        for method, ranking in rankings.items():
            ranked = ranking.loc[row_id]
            score = float(ranked["score"])
            cutoff = summaries[method]["arrival"]["cutoff"]
            reviewed = row_id in admissions[method]
            decision = "review" if reviewed else "below_cutoff" if score < cutoff else "capacity_full"
            observations[method] = {"score": score, "daily_rank": int(ranked["rank"]),
                                    "cutoff": cutoff, "decision": decision, "reviewed": reviewed,
                                    "batch_reviewed": bool(ranked["rank"] <= FIXED_BUDGET)}
        records.append({"row_id": row_id, "groups": categories, "time_seconds": float(row["Time"]),
                        "batch": int(row["day"]), "amount": float(row["Amount"]), "label": int(row["TX_FRAUD"]),
                        "features": {name: float(row[name]) for name in FEATURES},
                        "feature_context": [{"feature": name, "value": float(row[name]),
                                             "train_mean": float(center[name]), "train_std": float(spread[name])}
                                            for name in top_features],
                        "methods": observations})
    return records


def run():
    raw = read_source()
    raw["TRANSACTION_ID"] = raw.index
    raw["TX_FRAUD"] = raw["Class"]
    raw["log_amount"] = np.log1p(raw["Amount"])
    raw["day"] = (raw["Time"] // 21600).astype(int)
    train, dev, hold = raw[raw["day"] < 4], raw[raw["day"].between(4, 5)], raw[raw["day"] >= 6]
    print("Dev selection: six configurations x three seeds", flush=True)
    dev_models, grid = {}, []
    slots = FIXED_BUDGET * dev["day"].nunique()
    for config in GRID:
        counts, runs = [], []
        for seed in SEEDS:
            model = forest(config, seed).fit(train[FEATURES])
            scores = -model.score_samples(dev[FEATURES])
            cutoff = fit_cutoff(scores, slots)
            summary, _, _ = evaluate(dev, scores, cutoff)
            tp = summary["metrics"]["budgets"][FIXED_BUDGET]["tp"]
            counts.append(tp)
            runs.append({"seed": seed, **summary})
            dev_models[(config, seed)] = (model, scores, cutoff)
        row = {"config": {"max_samples": config[0], "max_features": config[1], "n_estimators": 100},
               "mean_dev_caught": float(np.mean(counts)), "std_dev_caught": float(np.std(counts)), "runs": runs}
        grid.append(row)
        print(config_name(config), counts, flush=True)
    winner = min(grid, key=lambda row: (-row["mean_dev_caught"], row["std_dev_caught"],
                                        row["config"]["max_samples"], row["config"]["max_features"]))
    selected = (winner["config"]["max_samples"], winner["config"]["max_features"])
    selection = {"rule": "mean dev Top50 catches desc, seed standard deviation asc, samples asc, features asc",
                 "chosen_config": winner["config"], "grid": grid, "seeds": SEEDS}
    # This file is written before the selected configuration scores holdout.
    save_json(ROOT / "if-dev-selection.json", selection)
    models = {}
    original = (256, 1.0)
    for seed in SEEDS:
        for name, config in (("original_if", original), ("selected_if", selected)):
            model, dev_scores, cutoff = dev_models[(config, seed)]
            models[f"{name}_{seed}"] = {"model": model, "dev_scores": dev_scores, "cutoff": cutoff}

    base_model = classifier().fit(train[FEATURES], train["Class"])
    base_dev = base_model.decision_function(dev[FEATURES])
    models["logistic"] = {"model": base_model, "dev_scores": base_dev, "cutoff": fit_cutoff(base_dev, slots)}
    folds_hash = None
    for seed in SEEDS:
        print(f"OOF IF + classifier, seed {seed}", flush=True)
        oof, folds = crossfit_scores(train[FEATURES], selected, seed)
        folds_hash = sha256(folds.tobytes()).hexdigest()
        model = classifier().fit(augmented(train[FEATURES], oof), train["Class"])
        dev_scores = model.decision_function(augmented(dev[FEATURES], models[f"selected_if_{seed}"]["dev_scores"]))
        models[f"combined_{seed}"] = {"model": model, "dev_scores": dev_scores,
                                      "cutoff": fit_cutoff(dev_scores, slots)}

    print("Evaluating the previously inspected final 12 hours", flush=True)
    summaries, rankings, admissions, diagnostics, dev_diagnostics = {}, {}, {}, {}, {}
    for name, entry in models.items():
        if name.startswith("combined"):
            seed = int(name.rsplit("_", 1)[1])
            values = -models[f"selected_if_{seed}"]["model"].score_samples(hold[FEATURES])
            scores = entry["model"].decision_function(augmented(hold[FEATURES], values))
        elif name == "logistic":
            scores = entry["model"].decision_function(hold[FEATURES])
        else:
            scores = -entry["model"].score_samples(hold[FEATURES])
        summary, ranking, admitted = evaluate(hold, scores, entry["cutoff"])
        dev_result, dev_rank, dev_admitted = evaluate(dev, entry["dev_scores"], entry["cutoff"])
        dev_summary = dev_result["metrics"]
        summaries[name] = {**summary, "dev": dev_summary}
        rankings[name], admissions[name] = ranking, admitted
        diagnostics[name] = summarize_windows(hold, dev, entry["dev_scores"], scores, ranking, admitted)
        dev_diagnostics[name] = summarize_windows(dev, dev, entry["dev_scores"],
                                                entry["dev_scores"], dev_rank, dev_admitted)
        print(name, "batch", summary["metrics"]["budgets"][FIXED_BUDGET]["tp"],
              "arrival", summary["arrival"]["caught"], flush=True)

    comparisons = {}
    label_by_id = hold.set_index("TRANSACTION_ID")["TX_FRAUD"]
    for seed in SEEDS:
        comparisons[f"combined_{seed}_vs_logistic"] = paired_changes(
            label_by_id, admissions["logistic"], admissions[f"combined_{seed}"])
        comparisons[f"selected_if_{seed}_vs_original"] = paired_changes(
            label_by_id, admissions[f"original_if_{seed}"], admissions[f"selected_if_{seed}"])
    qualifies = all(summaries[f"combined_{seed}"]["arrival"]["caught"] > summaries["logistic"]["arrival"]["caught"]
                    and summaries[f"combined_{seed}"]["arrival"]["false_alerts"] <= summaries["logistic"]["arrival"]["false_alerts"]
                    for seed in SEEDS)
    original_rank = rankings["original_if_42"].reset_index().sort_values(["rank", "day", "TRANSACTION_ID"])
    groups = {
        "original_false_alerts": original_rank[(original_rank["rank"] <= FIXED_BUDGET) &
                                              (original_rank["TX_FRAUD"] == 0)]["TRANSACTION_ID"].head(20).tolist(),
        "original_missed_frauds": original_rank[(original_rank["rank"] > FIXED_BUDGET) &
                                               (original_rank["TX_FRAUD"] == 1)]["TRANSACTION_ID"].head(20).tolist(),
    }
    for name, pair in comparisons.items():
        groups[f"{name}_rescued"] = pair["rescued_ids"][:3]
        groups[f"{name}_regressed"] = pair["regressed_ids"][:3]
    still_missed = set(label_by_id.index[label_by_id == 1]) - admissions["logistic"] - admissions["combined_42"]
    groups["still_missed"] = sorted(map(int, still_missed))[:3]
    return {"status": "complete", "scope": "Exploratory research on previously inspected ULB/Worldline data",
            "archive_sha256": ARCHIVE_SHA256, "source_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
            "train_rows": len(train), "dev_rows": len(dev), "evaluation_rows": len(hold),
            "evaluation_fraud": int(hold["Class"].sum()), "features": FEATURES, "budget": FIXED_BUDGET,
            "splits_hours": {"train": [0, 24], "dev": [24, 36], "evaluation": [36, 48]},
            "selection": selection, "crossfit": {"folds": 5, "shuffle_seed": 42, "assignment_sha256": folds_hash,
                                               "rows_scored_once": len(train),
                                               "purpose": "IF score used to train the combined classifier excludes its own row"},
            "results": summaries, "paired_arrival_changes": comparisons,
            "diagnostics": diagnostics, "dev_diagnostics": dev_diagnostics,
            "recommendation": "combined_candidate" if qualifies else "keep_logistic",
            "decision_rule": "Combined must catch more fraud with no extra false alerts in arrival replay for all three seeds",
            "case_groups": groups, "cases": case_rows(hold, train, rankings, admissions, summaries, groups)}


if __name__ == "__main__":
    report = run()
    save_json(ROOT / "if-research-report.json", report)
    print(report["recommendation"], flush=True)
