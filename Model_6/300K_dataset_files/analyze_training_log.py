#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 analyze_training_log.py
-------------------------------------------------------------------------------
 Analysis and visualization of training logs (log.csv) for seismic inversion
 models (e.g. ModelCNN_fvs — f-Vs dispersion -> shear-wave velocity profile).

 The log.csv file must contain the following columns:
   run_id, model_name, run_started_at, epoch (format "N/M"), epoch_timestamp,
   train_loss, val_loss, is_best, learning_rate, epoch_duration_sec,
   num_params, batch_size, seed

 Usage:
   Edit the settings inside the `if __name__ == "__main__":` block at the
   bottom of this file (CSV_PATH, MIN_EPOCHS, ...), then press Run in PyCharm.

 Outputs:
   - A text summary of each run printed to the terminal
   - figures/<run_id>_training_report.png   (4-panel report per run)
   - figures/runs_comparison.png            (if several runs are analyzed)

 Author: AI Intern — Civil Engineering Laboratory, NYCU
===============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend (server / SSH sessions)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Global figure style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_TRAIN = "#1f77b4"   # blue
COLOR_VAL   = "#d62728"   # red
COLOR_LR    = "#2ca02c"   # green
COLOR_GAP   = "#9467bd"   # purple
COLOR_BEST  = "#ff7f0e"   # orange


# ----------------------------------------------------------------------------
# Data loading and preparation
# ----------------------------------------------------------------------------
def load_log(csv_path: Path) -> pd.DataFrame:
    """Load the CSV log and normalize column types."""
    df = pd.read_csv(csv_path)

    required = {"run_id", "epoch", "train_loss", "val_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # epoch "N/M" -> two numeric columns
    epoch_split = df["epoch"].astype(str).str.split("/", expand=True)
    df["epoch_num"] = pd.to_numeric(epoch_split[0], errors="coerce")
    if epoch_split.shape[1] > 1:
        df["epoch_total"] = pd.to_numeric(epoch_split[1], errors="coerce")
    else:
        df["epoch_total"] = np.nan

    for col in ("train_loss", "val_loss", "learning_rate",
                "epoch_duration_sec", "num_params", "batch_size"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_best" in df.columns:
        df["is_best"] = df["is_best"].astype(str).str.lower().eq("true")

    for col in ("run_started_at", "epoch_timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df.sort_values(["run_id", "epoch_num"]).reset_index(drop=True)


def summarize_run(run: pd.DataFrame) -> dict:
    """Compute the key statistics of a single run."""
    best_idx = run["val_loss"].idxmin()
    best = run.loc[best_idx]
    last = run.iloc[-1]

    total_sec = run["epoch_duration_sec"].sum() if "epoch_duration_sec" in run else np.nan
    gap_final = last["val_loss"] - last["train_loss"]

    return {
        "run_id": run["run_id"].iloc[0],
        "model": run.get("model_name", pd.Series(["?"])).iloc[0],
        "criterion": run.get("criterion", pd.Series(["?"])).iloc[0],
        "epochs_logged": len(run),
        "epochs_planned": int(run["epoch_total"].iloc[0]) if run["epoch_total"].notna().any() else None,
        "best_val_loss": best["val_loss"],
        "best_epoch": int(best["epoch_num"]),
        "train_at_best": best["train_loss"],
        "final_train_loss": last["train_loss"],
        "final_val_loss": last["val_loss"],
        "final_gap": gap_final,
        "gap_ratio": last["val_loss"] / last["train_loss"] if last["train_loss"] > 0 else np.nan,
        "total_time_h": total_sec / 3600 if np.isfinite(total_sec) else np.nan,
        "mean_epoch_sec": run["epoch_duration_sec"].mean() if "epoch_duration_sec" in run else np.nan,
        "num_params": int(run["num_params"].iloc[0]) if "num_params" in run else None,
        "batch_size": int(run["batch_size"].iloc[0]) if "batch_size" in run else None,
        "lr_start": run["learning_rate"].iloc[0] if "learning_rate" in run else np.nan,
        "lr_end": run["learning_rate"].iloc[-1] if "learning_rate" in run else np.nan,
    }


def print_summary(stats: dict) -> None:
    """Print a human-readable summary of a run to the terminal."""
    line = "=" * 78
    print(line)
    print(f" RUN: {stats['run_id']}")
    print(line)
    planned = f"/{stats['epochs_planned']}" if stats["epochs_planned"] else ""
    print(f"  Model               : {stats['model']}"
          f"  ({stats['num_params']:,} parameters, batch={stats['batch_size']})"
          if stats["num_params"] else f"  Model               : {stats['model']}")
    print(f"  Criterion           : {stats['criterion']}")
    print(f"  Logged epochs       : {stats['epochs_logged']}{planned}")
    if stats["epochs_planned"] and stats["epochs_logged"] < stats["epochs_planned"]:
        print(f"  [i] Run stopped at epoch {stats['epochs_logged']}/{stats['epochs_planned']} "
              f"(early stopping or interrupted run — log.csv alone can't tell which).")
    print(f"  Best val loss       : {stats['best_val_loss']:.6f}  (epoch {stats['best_epoch']},"
          f" train={stats['train_at_best']:.6f})")
    print(f"  Final losses        : train={stats['final_train_loss']:.6f}"
          f"  |  val={stats['final_val_loss']:.6f}")
    print(f"  Final val-train gap : {stats['final_gap']:+.6f}"
          f"  (val/train ratio = {stats['gap_ratio']:.2f})")
    if np.isfinite(stats["total_time_h"]):
        print(f"  Total duration      : {stats['total_time_h']:.2f} h"
              f"  (~{stats['mean_epoch_sec']:.1f} s/epoch)")
    if np.isfinite(stats["lr_start"]):
        print(f"  Learning rate       : {stats['lr_start']:.2e}  ->  {stats['lr_end']:.2e}")

    # --- Automatic diagnosis ----------------------------------------------
    print("  Diagnosis:")
    ratio = stats["gap_ratio"]
    best_ep, n_ep = stats["best_epoch"], stats["epochs_logged"]
    if np.isfinite(ratio) and ratio > 3:
        print("    [!] Strong overfitting: val loss is >3x the train loss.")
        print("        -> Ideas: data augmentation (noise on seismograms,")
        print("           offset variation), dropout/weight decay, or a smaller model.")
    elif np.isfinite(ratio) and ratio > 1.5:
        print("    [~] Moderate overfitting (val > 1.5x train). Worth monitoring.")
    else:
        print("    [OK] Reasonable train/val gap.")

    if n_ep >= 20 and best_ep < 0.5 * n_ep:
        print(f"    [i] The best model dates back to epoch {best_ep}/{n_ep}:")
        print("        the second half of the training brought no improvement")
        print("        -> a more aggressive early stopping would save GPU time.")
    print()


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_run_report(run: pd.DataFrame, stats: dict, outdir: Path) -> Path:
    """4-panel report: loss, loss (log scale), learning rate + gap, epoch duration."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Training report — {stats['run_id']}", fontsize=14, y=0.98)
    ep = run["epoch_num"]

    # (1) Loss curves, linear scale -------------------------------------------
    ax = axes[0, 0]
    ax.plot(ep, run["train_loss"], color=COLOR_TRAIN, label="Train loss")
    ax.plot(ep, run["val_loss"], color=COLOR_VAL, label="Validation loss")
    ax.scatter([stats["best_epoch"]], [stats["best_val_loss"]],
               color=COLOR_BEST, zorder=5, s=60, marker="*",
               label=f"Best val ({stats['best_val_loss']:.4f} @ ep {stats['best_epoch']})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Learning curves")
    ax.legend()

    # (2) Loss curves, log scale ----------------------------------------------
    ax = axes[0, 1]
    ax.semilogy(ep, run["train_loss"], color=COLOR_TRAIN, label="Train loss")
    ax.semilogy(ep, run["val_loss"], color=COLOR_VAL, label="Validation loss")
    ax.axvline(stats["best_epoch"], color=COLOR_BEST, ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title("Learning curves (log scale)")
    ax.legend()

    # (3) Learning rate + val-train gap ---------------------------------------
    ax = axes[1, 0]
    if run["learning_rate"].notna().any():
        ax.semilogy(ep, run["learning_rate"], color=COLOR_LR, label="Learning rate")
        ax.set_ylabel("Learning rate (log)", color=COLOR_LR)
        ax.tick_params(axis="y", labelcolor=COLOR_LR)
    ax2 = ax.twinx()
    ax2.plot(ep, run["val_loss"] - run["train_loss"], color=COLOR_GAP, alpha=0.85,
             label="Val − train gap")
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("Val − train gap", color=COLOR_GAP)
    ax2.tick_params(axis="y", labelcolor=COLOR_GAP)
    ax2.spines["right"].set_visible(True)
    ax.set_xlabel("Epoch")
    ax.set_title("Learning rate & overfitting")

    # (4) Duration per epoch ---------------------------------------------------
    ax = axes[1, 1]
    if run["epoch_duration_sec"].notna().any():
        ax.plot(ep, run["epoch_duration_sec"], color="#555", lw=0.9)
        mean = run["epoch_duration_sec"].mean()
        ax.axhline(mean, color="#555", ls="--", lw=1,
                   label=f"Mean: {mean:.1f} s")
        ax.set_ylabel("Duration (s)")
        ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_title("Duration per epoch (hardware / I/O stability)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outpath = outdir / f"{stats['run_id']}_training_report.png"
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def plot_runs_comparison(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Compare the validation loss of all runs on a single figure."""
    runs = df["run_id"].unique()
    if len(runs) < 2:
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    for i, run_id in enumerate(runs):
        run = df[df["run_id"] == run_id]
        ax.plot(run["epoch_num"], run["val_loss"],
                color=cmap(i % 10), label=f"{run_id} ({len(run)} ep, {run['criterion'].iloc[0] if 'criterion' in run.columns else '?'})",)
        best = run.loc[run["val_loss"].idxmin()]
        ax.scatter([best["epoch_num"]], [best["val_loss"]],
                   color=cmap(i % 10), marker="*", s=80, zorder=5)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss")
    ax.set_title("Runs comparison — validation loss (★ = best point)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    outpath = outdir / "runs_comparison.png"
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main(csv_path: str | Path,
         run_id: str | None = None,
         min_epochs: int = 1,
         outdir: str | Path = "figures",
         exclude_models: list[str] | None = None) -> int:
    """Run the full analysis.

    Parameters
    ----------
    csv_path       : path to log.csv (relative paths are resolved from this script's folder)
    run_id         : analyze only this run_id, or None to analyze all runs
    min_epochs     : ignore runs with fewer epochs (filters out aborted test runs)
    outdir         : output directory for the figures
    exclude_models : list of model_name values to exclude from the analysis
                     (e.g. a model with very poor results that would skew the
                     comparison plots), or None to keep every model
    """
    # Relative paths are resolved from this script's folder, so the
    # PyCharm Run button works regardless of the working directory.
    script_dir = Path(__file__).resolve().parent
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = script_dir / csv_path
    outdir = Path(outdir)
    if not outdir.is_absolute():
        outdir = script_dir / outdir

    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        print("Hint: put log.csv next to this script, or edit CSV_PATH "
              "in the settings at the bottom of the file.", file=sys.stderr)
        return 1

    df = load_log(csv_path)
    outdir.mkdir(parents=True, exist_ok=True)

    if run_id:
        df = df[df["run_id"] == run_id]
        if df.empty:
            print(f"Error: run_id '{run_id}' not found.", file=sys.stderr)
            return 1

    if exclude_models:
        if "model_name" not in df.columns:
            print("Error: cannot apply exclude_models, "
                  "column 'model_name' not found in the log.", file=sys.stderr)
            return 1
        excluded_mask = df["model_name"].isin(exclude_models)
        if excluded_mask.any():
            print(f"[i] Excluding model(s): {', '.join(exclude_models)} "
                  f"({excluded_mask.sum()} rows removed)\n")
        df = df[~excluded_mask]
        if df.empty:
            print("No run left to analyze after excluding models.", file=sys.stderr)
            return 1

    # Filter out runs that are too short (aborted tests)
    counts = df.groupby("run_id")["epoch_num"].count()
    kept = counts[counts >= min_epochs].index
    skipped = counts[counts < min_epochs]
    if len(skipped):
        print(f"[i] {len(skipped)} run(s) skipped (< {min_epochs} epochs):")
        for rid, n in skipped.items():
            print(f"      - {rid} ({n} epochs)")
        print()
    df = df[df["run_id"].isin(kept)]

    if df.empty:
        print("No run left to analyze after filtering.", file=sys.stderr)
        return 1

    all_stats = []
    for rid in df["run_id"].unique():
        run = df[df["run_id"] == rid].reset_index(drop=True)
        stats = summarize_run(run)
        all_stats.append(stats)
        print_summary(stats)
        path = plot_run_report(run, stats, outdir)
        print(f"  Figure saved: {path}\n")

    comp = plot_runs_comparison(df, outdir)
    if comp:
        print(f"Comparison figure: {comp}")

    # Final summary table
    if len(all_stats) > 1:
        rec = pd.DataFrame(all_stats)[
            ["run_id", "criterion", "epochs_logged", "best_val_loss", "best_epoch",
             "final_gap", "total_time_h"]]
        print("\nSummary:")
        print(rec.to_string(index=False,
                            float_format=lambda x: f"{x:.4f}"))

    return 0


if __name__ == "__main__":
    # -- Settings
    CSV_PATH = "log/90K_RMSELoss_std0.1/log_all_models.csv"  # path to the log file (relative to this script)
    RUN_ID = None        # e.g. "ModelResNet50_fvs_20260707_032232", or None for all runs
    MIN_EPOCHS = 10           # ignore runs with fewer epochs (aborted tests).
    # Lowered from 100: with early_stopping_patience=25 in train.py, a legitimate
    # run can now stop as early as ~epoch 26 — a threshold of 100 would silently
    # filter those out along with actual aborted/crashed runs. Raise this back up
    # if you disable early stopping (early_stopping_patience=0) and want to filter
    # short test runs again.
    OUTDIR = "figures/90K_RMSELoss_std0.1"   # output directory for the figures
    EXCLUDE_MODELS = [
        "ModelEfficientNetB0_fvs_Noise_std0.08",
    ]  # model_name values to exclude from the analysis (bad results, etc.)

    # =========================================================================

    sys.exit(main(
        CSV_PATH,
        run_id=RUN_ID,
        min_epochs=MIN_EPOCHS,
        outdir=OUTDIR,
        # exclude_models=EXCLUDE_MODELS
    ))