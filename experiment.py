"""Train two anomaly scorers and replay a fixed daily review budget."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from hashlib import sha1, sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent
REVISION = "6e67dbd0a3bfe0d7ec33abc4bce5f37cd4ff0d6a"
SOURCE = "Fraud-Detection-Handbook/simulated-data-raw"
DAYS = [(date(2018, 4, 1) + timedelta(days=i)).isoformat() for i in range(35)]
BUDGETS = [10, 25, 50, 100, 200]
FIXED_BUDGET = 50
SEED = 42
RAW = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"]
FEATURES = ["log_amount", "prior_tx_1h", "prior_tx_24h", "prior_tx_total",
            "amount_ratio_to_history", "new_terminal"]
METHODS = {"amount": "Amount baseline", "amount_if": "Isolation Forest · amount",
           "behavior_if": "Isolation Forest · behavior"}
DATA_DIR = ROOT / "data"


def fetch(url):
    with urlopen(url, timeout=30) as response:
        return response.read()


def load_dataset():
    manifest = DATA_DIR / "source-tree.json"
    DATA_DIR.mkdir(exist_ok=True)
    if not manifest.exists():
        api = f"https://api.github.com/repos/{SOURCE}/git/trees/{REVISION}?recursive=1"
        manifest.write_bytes(fetch(api))
    tree = json.loads(manifest.read_bytes())
    if tree.get("sha") != REVISION:
        raise ValueError("Unrecognized revision")
    blobs = {item["path"]: item["sha"] for item in tree["tree"]}

    def load_day(day):
        name = f"{day}.pkl"
        path = DATA_DIR / name
        blob_path = f"data/{name}"
        content = path.read_bytes() if path.exists() else fetch(
            f"https://raw.githubusercontent.com/{SOURCE}/{REVISION}/{blob_path}"
        )
        git_header = f"blob {len(content)}\0".encode("ascii")
        if sha1(git_header + content).hexdigest() != blobs[blob_path]:
            raise ValueError(f"Checksum mismatch for {day}")
        if not path.exists():
            path.write_bytes(content)
        # Parse only pinned first-party dataset files verified against their Git blob IDs.
        frame = pd.read_pickle(BytesIO(content))
        return frame, sha256(content).hexdigest()

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(load_day, DAYS))
    frame = pd.concat([frame for frame, _ in files], ignore_index=True)
    frame = frame.sort_values(["TX_DATETIME", "TRANSACTION_ID"], ignore_index=True)
    frame["day"] = frame["TX_DATETIME"].dt.strftime("%Y-%m-%d")
    if not frame["TRANSACTION_ID"].is_unique or set(frame["day"]) != set(DAYS):
        raise ValueError("Source contains duplicate IDs or unexpected dates")
    return frame, dict(zip(DAYS, (digest for _, digest in files)))


def feature_rows(events):
    """Use only the current event fields and history strictly earlier than its timestamp."""
    rows = events[RAW].copy()
    amounts = rows["TX_AMOUNT"].to_numpy(dtype=float)
    seconds = rows["TX_DATETIME"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    prior_1h = np.zeros(len(rows), dtype=int)
    prior_24h = np.zeros(len(rows), dtype=int)
    prior_total = np.zeros(len(rows), dtype=int)
    previous_amount = np.full(len(rows), np.nan)
    for idx in rows.groupby("CUSTOMER_ID", sort=False).indices.values():
        t = seconds[idx]
        if np.any(t[:-1] > t[1:]):
            raise ValueError("Events must be sorted by time")
        prior = t.searchsorted(t, side="left")
        prior_1h[idx] = prior - t.searchsorted(t - 3600, side="left")
        prior_24h[idx] = prior - t.searchsorted(t - 86400, side="left")
        prior_total[idx] = prior
        subtotal = np.r_[0.0, np.cumsum(amounts[idx])]
        previous_amount[idx] = np.divide(subtotal[prior], prior,
                                        out=np.full(len(idx), np.nan), where=prior > 0)
    first_terminal = rows.groupby(["CUSTOMER_ID", "TERMINAL_ID"])["TX_DATETIME"].transform("min")
    rows["log_amount"] = np.log1p(amounts)
    rows["prior_tx_1h"] = prior_1h
    rows["prior_tx_24h"] = prior_24h
    rows["prior_tx_total"] = prior_total
    rows["prior_mean_amount"] = previous_amount
    rows["amount_ratio_to_history"] = np.divide(
        amounts, previous_amount, out=np.ones(len(rows)), where=previous_amount > 0
    )
    rows["new_terminal"] = (rows["TX_DATETIME"] == first_terminal).astype(int)
    return rows


def fit_scores(train, evaluation):
    scores = {"amount": evaluation["TX_AMOUNT"].to_numpy()}
    for key, columns in [("amount_if", ["log_amount"]), ("behavior_if", FEATURES)]:
        model = IsolationForest(n_estimators=100, max_samples=256, random_state=SEED, n_jobs=2)
        model.fit(train[columns])
        scores[key] = -model.score_samples(evaluation[columns])
    return scores


def rank_and_measure(rows, scores):
    ranked = rows[["day", "TRANSACTION_ID", "TX_FRAUD"]].copy()
    ranked["score"] = scores
    ranked = ranked.sort_values(["day", "score", "TRANSACTION_ID"], ascending=[True, False, True])
    ranked["rank"] = ranked.groupby("day").cumcount() + 1
    daily = ranked.groupby("day")["TX_FRAUD"].agg(["size", "sum"])
    budgets = {}
    for budget in BUDGETS:
        workload = ranked[ranked["rank"] <= budget].groupby("day")["TX_FRAUD"].agg(["size", "sum"])
        tp, reviewed, fraud = int(workload["sum"].sum()), int(workload["size"].sum()), int(daily["sum"].sum())
        budgets[budget] = {"tp": tp, "fp": reviewed - tp, "fn": fraud - tp, "reviewed": reviewed,
                           "precision": tp / reviewed, "recall": tp / fraud, "days": [
            {"day": str(day), "transactions": int(daily.loc[day, "size"]), "fraud": int(daily.loc[day, "sum"]),
             "reviewed": int(row["size"]), "caught": int(row["sum"])}
            for day, row in workload.iterrows()
        ]}
    return ranked, {"ap": float(average_precision_score(rows["TX_FRAUD"], scores)), "budgets": budgets}


def casebook(raw, features, ranks):
    test = raw.set_index("TRANSACTION_ID")
    ledger = pd.DataFrame({m: rank.set_index("TRANSACTION_ID")["rank"] for m, rank in ranks.items()})
    ledger["fraud"] = test["TX_FRAUD"]
    ledger["scenario"] = test["TX_FRAUD_SCENARIO"]
    a, b, fraud = ledger["amount"], ledger["behavior_if"], ledger["fraud"].astype(bool)
    selectors = {
        "behavior_only": fraud & (b <= FIXED_BUDGET) & (a > FIXED_BUDGET),
        "amount_only": fraud & (a <= FIXED_BUDGET) & (b > FIXED_BUDGET),
        "false_positive": ~fraud & (b <= FIXED_BUDGET),
        "missed": fraud & (a > FIXED_BUDGET) & (b > FIXED_BUDGET),
    }
    result = {}
    lookup = features.set_index("TRANSACTION_ID")
    scores = {m: rows.set_index("TRANSACTION_ID")["score"] for m, rows in ranks.items()}
    for kind, mask in selectors.items():
        eligible = ledger[mask].sort_values("behavior_if" if kind != "amount_only" else "amount")
        if eligible.empty:
            continue
        transaction = int(eligible.index[0])
        event, observed = test.loc[transaction], lookup.loc[transaction]
        history = features[(features["CUSTOMER_ID"] == event["CUSTOMER_ID"]) &
                           (features["TX_DATETIME"] < event["TX_DATETIME"])].tail(10)
        result[kind] = {
            "event": {"transaction_id": transaction, "timestamp": str(event["TX_DATETIME"]),
                      "customer_id": int(event["CUSTOMER_ID"]), "terminal_id": int(event["TERMINAL_ID"]),
                      "amount": float(event["TX_AMOUNT"])},
            "features": {key: round(float(observed[key]), 4) for key in FEATURES},
            "prior_mean_amount": None if pd.isna(observed["prior_mean_amount"])
            else round(float(observed["prior_mean_amount"]), 2),
            "history": [{"timestamp": str(row.TX_DATETIME), "terminal": int(row.TERMINAL_ID),
                         "amount": float(row.TX_AMOUNT)} for row in history.itertuples(index=False)],
            "scores": {m: {"score": round(float(scores[m].loc[transaction]), 6),
                           "daily_rank": int(ledger.loc[transaction, m])} for m in METHODS},
            "label": int(event["TX_FRAUD"]), "scenario": int(event["TX_FRAUD_SCENARIO"]),
        }
    return result


def paired_outcomes(ranks):
    baseline = ranks["amount"].set_index("TRANSACTION_ID")
    behavior = ranks["behavior_if"].set_index("TRANSACTION_ID")
    fraud = baseline["TX_FRAUD"].astype(bool)
    a = baseline["rank"] <= FIXED_BUDGET
    b = behavior["rank"].reindex(baseline.index) <= FIXED_BUDGET
    return {"both": int((fraud & a & b).sum()), "amount_only": int((fraud & a & ~b).sum()),
            "behavior_only": int((fraud & ~a & b).sum()), "neither": int((fraud & ~a & ~b).sum())}


def experiment():
    print("Loading pinned public simulator data", flush=True)
    raw, checksums = load_dataset()
    print(f"Loaded {len(raw):,} transactions. Building point-in-time features.", flush=True)
    features = feature_rows(raw)
    splits = {"warmup": DAYS[:7], "train": DAYS[7:21], "dev": DAYS[21:28], "test": DAYS[28:]}
    train = features.loc[raw["day"].isin(splits["train"])]
    evaluation = raw["day"].isin(splits["dev"] + splits["test"])
    prediction_rows = raw[evaluation]
    print("Scoring both Isolation Forests", flush=True)
    scores = fit_scores(train, features[evaluation])
    results = {}
    for split in ("dev", "test"):
        mask = prediction_rows["day"].isin(splits[split]).to_numpy()
        selected = prediction_rows[mask]
        summary, ranks = {}, {}
        for method, values in scores.items():
            ranks[method], summary[method] = rank_and_measure(selected, values[mask])
        results[split] = {"rows": len(selected), "fraud": int(selected["TX_FRAUD"].sum()), "methods": summary}
        if split == "dev":
            dev_winner = max(METHODS, key=lambda name: summary[name]["budgets"][FIXED_BUDGET]["tp"])
    return {"source": SOURCE, "revision": REVISION, "daily_source_sha256": checksums,
            "days": splits, "rows": len(raw), "train_rows": len(train), "features": FEATURES,
            "methods": METHODS, "fixed_budget": FIXED_BUDGET, "budgets": BUDGETS,
            "dev_winner": dev_winner, "test_paired_at_fixed_budget": paired_outcomes(ranks),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
            "seed": SEED, "results": results, "cases": casebook(selected, features, ranks)}


def build_page(report, output):
    public_benchmark = ROOT / "kaggle-report.json"
    if public_benchmark.exists():
        report = {**report, "kaggle": json.loads(public_benchmark.read_text(encoding="utf-8"))}
    research = ROOT / "if-research-report.json"
    if research.exists():
        report = {**report, "if_research": json.loads(research.read_text(encoding="utf-8"))}
    queue = ROOT / "xgb-queue-report.json"
    if queue.exists():
        report = {**report, "queue_research": json.loads(queue.read_text(encoding="utf-8"))}
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = (ROOT / "lab.html").read_text(encoding="utf-8")
    template = template.replace("__RESEARCH_SCRIPT__", (ROOT / "research.js").read_text(encoding="utf-8"))
    template = template.replace("__QUEUE_SCRIPT__", (ROOT / "queue.js").read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__REPORT_JSON__", payload), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true", help="Rebuild the web lab from the existing report")
    parser.add_argument("--output", type=Path, default=ROOT / "index.html")
    args = parser.parse_args()
    report_path = ROOT / "report.json"
    if args.render_only:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = experiment()
        report_path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    build_page(report, args.output)


if __name__ == "__main__":
    main()
