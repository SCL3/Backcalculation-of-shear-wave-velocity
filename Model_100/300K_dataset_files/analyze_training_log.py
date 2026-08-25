#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
This code was generated using Claude for analyzing the csv traning log file.
 analyze_training_log.py
-------------------------------------------------------------------------------
 Analysis and visualization of training logs (log.csv) for seismic inversion
 models

 The log.csv file must contain the following columns:
   run_id, model_name, run_started_at, epoch (format "N/M"), epoch_timestamp,
   train_loss, val_loss, is_best, learning_rate, epoch_duration_sec,
   num_params, batch_size, seed

 Outputs:
   - A text summary of each run printed to the terminal
   - figures/<run_id>_training_report.png   (4-panel report per run)
   - figures/runs_comparison.png            (if several runs are analyzed)
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
    # Overfitting is measured at the BEST-VAL epoch, not the final epoch: it is
    # the checkpoint that actually gets deployed (is_best=True), and training
    # loss keeps dropping after that point while val loss has already plateaued
    # — so a final/final ratio overstates the gap of the model you actually use.
    gap_at_best = best["val_loss"] - best["train_loss"]

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
        "gap_at_best": gap_at_best,
        "gap_ratio": best["val_loss"] / best["train_loss"] if best["train_loss"] > 0 else np.nan,
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
          f"  |  val={stats['final_val_loss']:.6f}"
          f"  (gap={stats['final_gap']:+.6f})")
    print(f"  At best epoch ({stats['best_epoch']:>3d})  : train={stats['train_at_best']:.6f}"
          f"  |  val={stats['best_val_loss']:.6f}"
          f"  (gap={stats['gap_at_best']:+.6f}, val/train ratio = {stats['gap_ratio']:.2f})")
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
        print("    [!] Strong overfitting: at the best epoch, val loss is >3x train loss.")
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


def compute_shared_axis_bounds(df: pd.DataFrame, pad_frac: float = 0.04) -> dict:
    """Compute x/y limits shared by every run report generated in this session.

    epoch_max : the longest run sets the x-axis, so a run that stopped early
                (fewer epochs, e.g. by early stopping) simply shows its curve
                ending partway across an axis that matches every other report.
    loss_min/max : global min/max of train_loss and val_loss across ALL runs,
                so the same y-range is used everywhere. loss_min_log guards
                against a non-positive lower bound, which set_ylim would
                reject on a log-scale axis.
    """
    epoch_max = float(df["epoch_num"].max()) * (1 + pad_frac)
    loss_all = pd.concat([df["train_loss"], df["val_loss"]]).dropna()
    loss_min = float(loss_all.min())
    loss_max = float(loss_all.max()) * (1 + pad_frac)
    loss_span = loss_max - loss_min
    return {
        "epoch_max": epoch_max,
        "loss_min": loss_min - pad_frac * loss_span,
        "loss_min_log": max(loss_min * 0.9, 1e-6),
        "loss_max": loss_max,
    }


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_run_report(run: pd.DataFrame, stats: dict, outdir: Path,
                     axis_bounds: dict | None = None) -> Path:
    """4-panel report: loss, loss (log scale), learning rate + gap, epoch duration.

    axis_bounds, if given, forces the SAME x/y limits on the two loss panels
    across every run report generated in this session. Without it, matplotlib
    auto-scales each figure independently, so two reports placed side by side
    (e.g. in a slide) end up on different scales and are not visually
    comparable — a shorter run looks like it "starts higher" purely because
    its y-axis is tighter, not because its loss actually differs.
    """
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
    if axis_bounds:
        ax.set_xlim(0, axis_bounds["epoch_max"])
        ax.set_ylim(axis_bounds["loss_min"], axis_bounds["loss_max"])
    ax.legend()

    # (2) Loss curves, log scale ----------------------------------------------
    ax = axes[0, 1]
    ax.semilogy(ep, run["train_loss"], color=COLOR_TRAIN, label="Train loss")
    ax.semilogy(ep, run["val_loss"], color=COLOR_VAL, label="Validation loss")
    ax.axvline(stats["best_epoch"], color=COLOR_BEST, ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title("Learning curves (log scale)")
    if axis_bounds:
        ax.set_xlim(0, axis_bounds["epoch_max"])
        ax.set_ylim(axis_bounds["loss_min_log"], axis_bounds["loss_max"])
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
    """Compare the validation loss of all runs on a single figure.

    Runs are drawn worst-to-best (highest best-val-loss first), so the legend
    — which matplotlib fills in draw order for a single-column layout — lists
    the worst model at the top and the best at the bottom. That matches how
    the curves actually sit on the plot: worse (higher loss) runs plateau
    higher up, better (lower loss) runs plateau lower down, so legend order
    and curve position agree at a glance.
    """
    run_ids = df["run_id"].unique()
    if len(run_ids) < 2:
        return None

    ranking = sorted(run_ids, key=lambda rid: df.loc[df["run_id"] == rid, "val_loss"].min(),
                      reverse=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    for i, run_id in enumerate(ranking):
        run = df[df["run_id"] == run_id]
        ax.plot(run["epoch_num"], run["val_loss"],
                color=cmap(i % 10), label=f"{run_id} ({len(run)} ep, {run['criterion'].iloc[0] if 'criterion' in run.columns else '?'})",)
        best = run.loc[run["val_loss"].idxmin()]
        ax.scatter([best["epoch_num"]], [best["val_loss"]],
                   color=cmap(i % 10), marker="*", s=80, zorder=5)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss")
    ax.set_title("Runs comparison — validation loss (★ = best point, legend top→bottom = worst→best)")
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
    axis_bounds = compute_shared_axis_bounds(df)
    for rid in df["run_id"].unique():
        run = df[df["run_id"] == rid].reset_index(drop=True)
        stats = summarize_run(run)
        all_stats.append(stats)
        print_summary(stats)
        path = plot_run_report(run, stats, outdir, axis_bounds=axis_bounds)
        print(f"  Figure saved: {path}\n")

    comp = plot_runs_comparison(df, outdir)
    if comp:
        print(f"Comparison figure: {comp}")

    # Final summary table
    if len(all_stats) > 1:
        rec = pd.DataFrame(all_stats)[
            ["run_id", "criterion", "epochs_logged", "best_val_loss", "best_epoch",
             "final_gap", "total_time_h"]].sort_values("best_val_loss")
        print("\nSummary (best -> worst):")
        print(rec.to_string(index=False,
                            float_format=lambda x: f"{x:.4f}"))

    return 0


if __name__ == "__main__":
    # -- Settings
    CSV_PATH = "log/90K_RMSELoss_No_Noise/log_5K_all_models_v2.csv"  # path to the log file (relative to this script)
    RUN_ID = None        # e.g. "ModelResNet50_fvs_20260707_032232", or None for all runs
    MIN_EPOCHS = 10           # ignore runs with fewer epochs (aborted tests).
    OUTDIR = "figures/V2/5K_all_models_v2"   # output directory for the figures
    EXCLUDE_MODELS = [
        "ModelCustomCNN_fvs_No_Noise_90k_RMSELoss",
    ]  # model_name values to exclude from the analysis (bad results, etc.)

    # =========================================================================

    sys.exit(main(
        CSV_PATH,
        run_id=RUN_ID,
        min_epochs=MIN_EPOCHS,
        outdir=OUTDIR,
        exclude_models=EXCLUDE_MODELS
    ))