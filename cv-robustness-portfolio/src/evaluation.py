"""Metrics, tabular summaries, and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def train_linear_probe(train_features, train_labels, max_iter: int = 2000):
    """Train the same standardized multinomial logistic-regression probe per model."""
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        ),
    )
    classifier.fit(train_features, train_labels)
    return classifier


def metric_row(model: str, condition: str, severity: int, y_true, y_pred) -> dict:
    return {
        "model": model,
        "condition": condition,
        "severity": int(severity),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def finalize_metrics(rows: list[dict]) -> pd.DataFrame:
    """Add accuracy drop in percentage points relative to each model's clean score."""
    frame = pd.DataFrame(rows)
    clean = frame.loc[frame["condition"] == "clean", ["model", "accuracy"]]
    clean = clean.rename(columns={"accuracy": "clean_accuracy"})
    frame = frame.merge(clean, on="model", how="left")
    frame["accuracy_drop_pp"] = (frame["clean_accuracy"] - frame["accuracy"]) * 100.0
    frame["accuracy_pct"] = frame["accuracy"] * 100.0
    frame["clean_accuracy_pct"] = frame["clean_accuracy"] * 100.0
    return frame.sort_values(["model", "condition", "severity"]).reset_index(drop=True)


def save_summaries(metrics: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    corrupted = metrics[metrics["condition"] != "clean"]
    summary = metrics[metrics["condition"] == "clean"][["model", "accuracy_pct"]].rename(
        columns={"accuracy_pct": "clean_accuracy_pct"}
    )
    robust = corrupted.groupby("model", as_index=False).agg(
        mean_corrupted_accuracy_pct=("accuracy_pct", "mean"),
        mean_accuracy_drop_pp=("accuracy_drop_pp", "mean"),
    )
    summary = summary.merge(robust, on="model", how="left")
    summary.to_csv(output_dir / "summary.csv", index=False)

    for condition, group in corrupted.groupby("condition"):
        plt.figure(figsize=(7, 4.5))
        for model, model_group in group.groupby("model"):
            means = model_group.groupby("severity")["accuracy_pct"].mean().sort_index()
            plt.plot(means.index, means.values, marker="o", label=model)
        plt.xlabel("Severity")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy by severity: {condition}")
        plt.xticks(sorted(group["severity"].unique()))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"accuracy_by_severity_{condition}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 4.5))
    plot_data = corrupted.groupby(["model", "condition", "severity"], as_index=False)["accuracy_drop_pp"].mean()
    labels = [f"{r.condition}\nS{r.severity}" for r in plot_data.itertuples()]
    x = range(len(plot_data))
    plt.bar(x, plot_data["accuracy_drop_pp"])
    plt.xticks(list(x), [f"{r.model}\n{r.condition}-S{r.severity}" for r in plot_data.itertuples()], rotation=70, ha="right")
    plt.ylabel("Accuracy drop (percentage points)")
    plt.title("Accuracy drop relative to clean images")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_drop.png", dpi=160)
    plt.close()
