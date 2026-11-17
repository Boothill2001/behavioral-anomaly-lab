# Behavioral Anomaly Lab

**A fraud review queue has 50 slots every six hours. Which transactions should an analyst see, and what gets missed when that decision happens on arrival?**

I compared three scoring methods on **284,807 real, anonymized card transactions** from ULB and Worldline. In the held-out arrival replay, **logistic regression caught 69 of 115 frauds using 87 of 100 available reviews**. Isolation Forest caught zero using all 100 slots. [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · [Measured report](kaggle-report.json)

**Windows quick start.** Double-click `start.bat` to open the demo. The app runs offline and requires no Python installation. You can also [download the interactive HTML](https://raw.githubusercontent.com/Boothill2001/behavioral-anomaly-lab/main/index.html) and open it directly.

![The Kaggle experiment shows how the same models perform under a batch queue and an arrival-order queue.](kaggle-benchmark.png)

## One transaction explains the tradeoff

On arrival, **transaction 276864** scored **9.971**. The review cutoff, set on development data, was **10.554**. The queue skipped it. By the end of its batch the same transaction ranked **46th**, so a batch policy reviewing the top 50 would have included it. The historical label confirmed fraud.

| Input | Score | Queue decision | Outcome |
|---|---|---|---|
| Anonymized card transaction 276864 | 9.971 from logistic regression | Skip on arrival because 9.971 is below 10.554 | Missed fraud |

On the same holdout, waiting for each batch to finish caught **71 frauds with 100 reviews**. Deciding on arrival caught **69 with 87 reviews**. In the replay, the cutoff mattered even with spare queue capacity.

## Why logistic regression was selected

The amount baseline ranks by payment size. Isolation Forest learns an anomaly score from the inputs alone. Logistic regression uses the same inputs as Isolation Forest and also learns from confirmed labels. Both models score 28 anonymized components plus log amount.

| Method | Fraud caught on dev, batch queue | Fraud caught on holdout, batch queue | Fraud caught on holdout, arrival queue | False alerts, arrival queue |
|---|---:|---:|---:|---:|
| Amount baseline | 0/96 | 0/115 | 0/115 | 98 |
| Isolation Forest | 39/96 | 0/115 | 0/115 | 100 |
| **Class-weighted logistic regression** | **69/96** | **71/115** | **69/115** | **18** |

Logistic regression earned the highest development catch count at the fixed budget. Its holdout result supports carrying it forward when training labels are available. Isolation Forest's drop from 39 catches to zero motivates investigation. The demo shows score and feature distribution shifts, queue overflow, and confirmed catches by six-hour window. Two days of data cannot establish the cause of that drop or estimate production losses avoided.

## Follow account behavior through the pipeline

The Kaggle data has anonymized inputs and no account IDs. A [public transaction simulator](https://github.com/Fraud-Detection-Handbook/simulated-data-raw) supplies account history for the walkthrough. It shows the inputs and outputs of all five steps.

**Payment event → prior account history → anomaly score → review decision → evaluation**

Start with **transaction 285562**. Its payment was **2.37×** the account's prior average. The behavior model ranked it **10th**, compared with **145th** by amount. Toggle the scorer to see how it moves in and out of the 50-slot queue. Across seven held-out days, the amount baseline still caught **168 of 663** simulated frauds, versus **68 of 663** for behavioral Isolation Forest. The individual case shows where behavior helped. The cohort result explains why the baseline stayed the default on this simulator.

<details>
<summary>See the five-step walkthrough</summary>

![The simulator shows the selected transaction, the review decision, and holdout precision and recall.](benchmark.png)

</details>

## Evaluation and reproduction

<details>
<summary>Time splits, score cutoffs, and data checks</summary>

- **Kaggle.** The source includes 492 labeled frauds over two days. Fit on relative hours 0–24, select on 24–36, and evaluate on 36–48. Each six-hour batch allows up to 50 reviews. Set each arrival cutoff to the lowest score in the pooled development top 100. During replay, admit qualifying transactions in timestamp and row-ID order, subject to the six-hour cap. Historical training labels are assumed to have matured. The source lacks label-availability timestamps and analyst decisions. [Report and pinned archive checksum](kaggle-report.json)
- **Monitoring.** Compare development and holdout score and feature distributions using Jensen–Shannon divergence on development quantile bins. Review counts and eventual fraud catches come from the replay. This is retrospective monitoring.
- **Simulator.** Use 35 daily batches totaling 335,910 rows. The first 7 days warm up history, the next 14 fit the forests, the following 7 select the method, and the final 7 evaluate it. Feature history is strictly earlier than each event. The forests use 100 trees, 256 samples per tree, and seed 42. Source files are pinned and checked before parsing. [Report and source checksums](report.json)

</details>

Run with Python 3.11. Both datasets download on the first run.

```powershell
python -m pip install -r requirements.txt
python kaggle_benchmark.py
python experiment.py
python -m unittest discover -v
```

Open `index.html`. It embeds the reports and works offline. [GitHub Actions](https://github.com/Boothill2001/behavioral-anomaly-lab/actions/workflows/verify.yml) runs the unit tests and checks that rebuilding the demo from the committed reports produces the same HTML.
