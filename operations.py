"""Out-of-time queue replay and window diagnostics."""

import numpy as np


def fit_cutoff(dev_scores, slots):
    """Freeze the minimum score of the top N development events."""
    if not 0 < slots <= len(dev_scores):
        raise ValueError("Invalid calibration budget")
    return float(np.partition(dev_scores, -slots)[-slots])


def replay_queue(events, scores, cutoff, capacity):
    """Admit events in arrival order with a fixed cutoff and per-batch cap."""
    if capacity < 1 or len(events) != len(scores):
        raise ValueError("Invalid capacity or score count")
    ordered = events[["day", "Time", "TRANSACTION_ID", "TX_FRAUD"]].assign(score=scores)
    ordered = ordered.sort_values(["Time", "TRANSACTION_ID"])
    selected_ids = set()
    batches = []
    for batch, events_in_batch in ordered.groupby("day", sort=True):
        above_cutoff = events_in_batch[events_in_batch["score"] >= cutoff]
        selected = above_cutoff.head(capacity)
        selected_ids.update(selected["TRANSACTION_ID"])
        batches.append({"batch": int(batch), "rows": len(events_in_batch), "fraud": int(events_in_batch["TX_FRAUD"].sum()),
                        "above_cutoff": len(above_cutoff), "reviewed": len(selected),
                        "overflow": max(0, len(above_cutoff) - capacity),
                        "caught": int(selected["TX_FRAUD"].sum())})
    totals = {key: sum(r[key] for r in batches) for key in ("fraud", "above_cutoff", "reviewed", "overflow", "caught")}
    return {"cutoff": cutoff, **totals, "false_alerts": totals["reviewed"] - totals["caught"],
            "missed": totals["fraud"] - totals["caught"], "batches": batches}, selected_ids


def distribution_shift(reference, observed):
    """Jensen-Shannon divergence on reference quantile bins (base 2, range 0–1)."""
    # Include outer bins so out-of-range observations still contribute.
    inner = np.unique(np.quantile(reference, np.arange(.1, 1, .1)))
    bins = np.r_[-np.inf, inner, np.inf]
    p = np.histogram(reference, bins=bins)[0] / len(reference)
    q = np.histogram(observed, bins=bins)[0] / len(observed)
    m = (p + q) / 2

    def divergence(distribution):
        present = distribution > 0
        return np.sum(distribution[present] * np.log2(distribution[present] / m[present]))

    return float(np.clip((divergence(p) + divergence(q)) / 2, 0, 1))
