"""Study label-guided feature selection and normal-only IF training."""

from hashlib import sha256

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import mutual_info_classif

from experiment import ROOT, rank_and_measure
from if_research import forest, save_json
from kaggle_benchmark import ARCHIVE_SHA256, FEATURES, read_source
from queue_experiment import calibrate, evaluate, make_xgb, reviewed_ids
from operations import fit_cutoff

SEEDS = (42, 43, 44)
SIZES = (5, 10, 15, 29)


def select_features(train):
    """Use training labels only, with a deterministic tie-break."""
    values = mutual_info_classif(train[FEATURES], train["Class"], random_state=42, n_neighbors=3)
    return sorted(zip(FEATURES, values), key=lambda pair: (-pair[1], pair[0]))


def configuration_key(row, branch):
    if branch == "standalone":
        values = [run["standalone"]["caught"] for run in row["dev_runs"]]
        false = [run["standalone"]["false_alerts"] for run in row["dev_runs"]]
    else:
        values = [run["reserved"]["branches"]["if"]["caught"] for run in row["dev_runs"]]
        false = [run["reserved"]["branches"]["if"]["false_alerts"] for run in row["dev_runs"]]
    return (-float(np.mean(values)), float(np.mean(false)), float(np.std(values)),
            len(row["features"]), row["normal_only"], row["name"])


def compare_ids(labels, baseline_ids, candidate_ids):
    fraud = set(labels.index[labels == 1])
    return {"rescued_ids": sorted(map(int, fraud & (candidate_ids - baseline_ids))),
            "regressed_ids": sorted(map(int, fraud & (baseline_ids - candidate_ids)))}


def run():
    raw = read_source()
    raw["TRANSACTION_ID"], raw["TX_FRAUD"] = raw.index, raw["Class"]
    raw["log_amount"] = np.log1p(raw["Amount"])
    raw["day"] = (raw["Time"] // 21600).astype(int)
    train, dev, held = raw[raw["day"] < 4], raw[raw["day"].between(4, 5)], raw[raw["day"] >= 6]
    print("Computing feature relevance using train labels only", flush=True)
    relevance = select_features(train)
    main_models = {seed: make_xgb(4, False, 1, seed).fit(train[FEATURES], train["Class"]) for seed in SEEDS}
    dev_main = {seed: model.predict_proba(dev[FEATURES])[:, 1] for seed, model in main_models.items()}
    models, candidates = {}, []
    normal = train[train["Class"] == 0]
    for normal_only in (False, True):
        fitted_rows = normal if normal_only else train
        for size in SIZES:
            columns = FEATURES if size == 29 else [column for column, _ in relevance[:size]]
            name = f"{'normal' if normal_only else 'all'}_top{size}"
            row = {"name": name, "normal_only": normal_only, "features": columns,
                   "training_rows": len(fitted_rows), "dev_runs": []}
            for seed in SEEDS:
                model = forest((1024, 1.0), seed).fit(fitted_rows[columns])
                values = -model.score_samples(dev[columns])
                standalone_cutoff = fit_cutoff(values, 100)
                # Use the same pure arrival router for IF-alone and the two-branch policy.
                standalone, _ = evaluate(dev, values, values, (standalone_cutoff, None), (50, 0))
                reserved_cutoffs = calibrate(dev_main[seed], values, (45, 5))
                reserved, _ = evaluate(dev, dev_main[seed], values, reserved_cutoffs, (45, 5))
                _, batch = rank_and_measure(dev, values)
                row["dev_runs"].append({"seed": seed, "standalone": standalone, "reserved": reserved,
                                        "batch_top50": batch["budgets"][50],
                                        "standalone_cutoff": standalone_cutoff,
                                        "reserved_cutoffs": reserved_cutoffs})
                models[(name, seed)] = (model, values)
            candidates.append(row)
            print(name, "dev standalone", [x["standalone"]["caught"] for x in row["dev_runs"]],
                  "extra IF", [x["reserved"]["branches"]["if"]["caught"] for x in row["dev_runs"]], flush=True)
    independent = min(candidates, key=lambda row: configuration_key(row, "standalone"))
    secondary = min(candidates, key=lambda row: configuration_key(row, "secondary"))
    main_dev = {}
    for seed in SEEDS:
        scores = dev_main[seed]
        cutoff = fit_cutoff(scores, 100)
        main_dev[str(seed)], _ = evaluate(dev, scores, scores, (cutoff, None), (50, 0))
    passes = all(
        run["reserved"]["caught"] > main_dev[str(run["seed"])]["caught"] and
        run["reserved"]["false_alerts"] <= main_dev[str(run["seed"])]["false_alerts"]
        for run in secondary["dev_runs"]
    )
    selection = {"standalone": independent["name"], "secondary": secondary["name"],
                 "policy": "reserved_45_5" if passes else "main_50",
                 "standalone_rule": "dev arrival catches desc, false alerts asc, seed std asc, feature count asc, all-training first",
                 "secondary_rule": "dev IF-branch catches desc, IF false alerts asc, seed std asc, feature count asc, all-training first",
                 "policy_rule": "reserved must catch more fraud with no extra false alerts in every seed on dev",
                 "train_feature_relevance": [{"feature": name, "mutual_information": float(value)} for name, value in relevance],
                 "candidates": candidates, "main_dev": main_dev}
    save_json(ROOT / "if-feature-selection.json", selection)
    print("Frozen:", independent["name"], secondary["name"], selection["policy"], flush=True)

    results, traces, comparisons = {}, {}, {}
    all_columns = {row["name"]: row["features"] for row in candidates}
    # Score every fixed candidate for transparent ablation, after recording selection.
    for candidate in candidates:
        name, columns = candidate["name"], candidate["features"]
        for seed in SEEDS:
            model, _ = models[(name, seed)]
            values = -model.score_samples(held[columns])
            settings = next(run for run in candidate["dev_runs"] if run["seed"] == seed)
            single, ledger = evaluate(held, values, values, (settings["standalone_cutoff"], None), (50, 0))
            _, batch = rank_and_measure(held, values)
            results[f"{name}_{seed}"] = {"standalone": single, "batch": batch,
                                        "cutoff": settings["standalone_cutoff"]}
            if name in (independent["name"], "all_top29"):
                traces[f"{name}_{seed}"] = ledger
    for seed in SEEDS:
        model_main_scores = main_models[seed].predict_proba(held[FEATURES])[:, 1]
        cutoff = fit_cutoff(dev_main[seed], 100)
        main_result, traces[f"main_50_{seed}"] = evaluate(
            held, model_main_scores, model_main_scores, (cutoff, None), (50, 0))
        settings = next(run for run in secondary["dev_runs"] if run["seed"] == seed)
        model = models[(secondary["name"], seed)][0]
        values = -model.score_samples(held[all_columns[secondary["name"]]])
        reserved, traces[f"reserved_45_5_{seed}"] = evaluate(
            held, model_main_scores, values, settings["reserved_cutoffs"], (45, 5))
        results[f"main_50_{seed}"] = {"arrival": main_result, "cutoffs": [cutoff, None]}
        results[f"reserved_45_5_{seed}"] = {"arrival": reserved, "cutoffs": settings["reserved_cutoffs"]}
        labels = held.set_index("TRANSACTION_ID")["TX_FRAUD"]
        comparisons[f"standalone_{seed}"] = compare_ids(
            labels, reviewed_ids(traces[f"all_top29_{seed}"]), reviewed_ids(traces[f"{independent['name']}_{seed}"]))
        comparisons[f"reserved_{seed}"] = compare_ids(
            labels, reviewed_ids(traces[f"main_50_{seed}"]), reviewed_ids(traces[f"reserved_45_5_{seed}"]))
        print(seed, "IF", results[f"{independent['name']}_{seed}"]["standalone"]["caught"],
              "IF branch", reserved["branches"]["if"]["caught"], "reserved total", reserved["caught"], flush=True)

    groups = {}
    fraud_ids = set(held.index[held["Class"] == 1])
    for seed in SEEDS:
        for prefix in ("standalone", "reserved"):
            change = comparisons[f"{prefix}_{seed}"]
            groups[f"{prefix}_rescued_{seed}"] = change["rescued_ids"][:3]
            groups[f"{prefix}_regressed_{seed}"] = change["regressed_ids"][:3]
        reserved_trace = traces[f"reserved_45_5_{seed}"]
        if_admitted = set(reserved_trace.index[(reserved_trace["branch"] == "if") &
                                             (reserved_trace["decision"] == "review")])
        main_admitted = reviewed_ids(traces[f"main_50_{seed}"])
        groups[f"if_unique_catches_{seed}"] = sorted(map(int, (if_admitted & fraud_ids) - main_admitted))[:3]
        groups[f"if_false_alerts_{seed}"] = sorted(map(int, if_admitted - fraud_ids))[:3]
    all_ids = sorted({row_id for ids in groups.values() for row_id in ids})
    cases = []
    for row_id in all_ids:
        row = held.loc[row_id]
        observations = {}
        for key, ledger in traces.items():
            observation = ledger.loc[row_id].to_dict()
            if not key.startswith("reserved"):
                observation["if_score"] = None
            observations[key] = observation
        cases.append({"row_id": row_id, "label": int(row["Class"]), "amount": float(row["Amount"]),
                      "time_seconds": float(row["Time"]),
                      "features": {column: float(row[column]) for column in FEATURES}, "traces": observations})
    return {"status": "complete", "scope": "Exploratory, previously inspected two-day data",
            "archive_sha256": ARCHIVE_SHA256,
            "source_code_sha256": sha256((ROOT / "if_feature_study.py").read_bytes()).hexdigest(),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
            "seeds": SEEDS, "selection": selection, "results": results, "comparisons": comparisons,
            "case_groups": groups, "cases": cases,
            "methodology": "Feature selection uses train labels; normal-only uses confirmed train negatives. Both are label-informed.",
            "evaluation_fraud": int(held["Class"].sum()), "evaluation_rows": len(held),
            "train_rows": len(train), "normal_training_rows": len(normal),
            "splits_hours": {"train": [0, 24], "dev": [24, 36], "evaluation": [36, 48]}}


if __name__ == "__main__":
    save_json(ROOT / "if-feature-report.json", run())
