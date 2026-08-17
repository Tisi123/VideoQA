"""
Plot validation accuracy/loss curves and per-category accuracy curves for one or
more VideoQA training runs, reading directly from the train_log.txt / evaluate.txt
files each run already produces.

Usage:
    python plot_results.py

Edit RUNS below to point at your result directories. Each entry needs:
    - a short display name (used in legends)
    - the path to that run's output_dir (the one containing train_log.txt and evaluate.txt)

Produces two PNGs in --out_dir:
    a) accuracy_and_loss.png -- val_acc (top) and val_loss (bottom) vs epoch, one line
       per run, so you can compare configs directly.
    b) categories_by_config.png -- ONE figure, one subplot per NExT-QA question
       category (CH/CW/TN/TC/DL/DC/DO/TP), each subplot showing one line per run
       (e.g. 1x1 vs 2x2 vs 3x3), so you can compare configs within each category.
"""

import os
import re
import json
import argparse
import matplotlib.pyplot as plt

# >>> ADAPT: point these at your actual result directories.
RUNS = [
    ("sevila", r"D:\Dokumente\UNI\MASTER\Masterarbeit\VideoQA\Resuls_SeViLA\sevila_base"),
    ("random_q", r"D:\Dokumente\UNI\MASTER\Masterarbeit\VideoQA\Resuls_SeViLA\random_qformer_ft"),
]

# NExT-QA question categories, in a fixed order so colors stay consistent across plots.
CATEGORIES = ["CH", "CW", "TN", "TC", "DL", "DC", "DO", "TP"]


def parse_train_log(path):
    """Parses lines like:
    epoch=0 step=8533 val_loss=0.4888 val_acc=0.4195 num_val=4996
    Returns a list of dicts, one per epoch, in file order.
    """
    pattern = re.compile(
        r"epoch=(\d+)\s+step=(\d+)\s+val_loss=([\d.]+)\s+val_acc=([\d.]+)\s+num_val=(\d+)"
    )
    rows = []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            epoch, step, val_loss, val_acc, num_val = m.groups()
            rows.append({
                "epoch": int(epoch),
                "step": int(step),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "num_val": int(num_val),
            })
    return rows


def parse_evaluate(path):
    """Parses one JSON object per line, e.g. {"val": {"agg_metrics": ..., "CH": ..., ...}}.
    Returns a list of the inner "val" dicts, in file order (== epoch order)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj["val"])
    return rows


def plot_accuracy_and_loss(runs_data, out_path):
    """Stacked subplots: val_acc on top, val_loss on bottom, one line per run."""
    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for name, rows in runs_data.items():
        epochs = [r["epoch"] for r in rows]
        acc = [r["val_acc"] for r in rows]
        loss = [r["val_loss"] for r in rows]
        ax_acc.plot(epochs, acc, marker="o", label=name)
        ax_loss.plot(epochs, loss, marker="o", label=name)

    ax_acc.set_ylabel("Validation accuracy")
    ax_acc.set_title("Validation accuracy vs. epoch")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    ax_loss.set_ylabel("Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_title("Validation loss vs. epoch")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_categories_grid(runs_eval_data, out_path):
    """One figure, one subplot per category (8 total), each subplot showing one
    line per run -- so you can directly compare 1x1 vs 2x2 vs 3x3 (etc.) within
    each question category.

    runs_eval_data: dict {run_name: eval_rows}, where eval_rows is the list
    returned by parse_evaluate() for that run.
    """
    n_cats = len(CATEGORIES)
    n_cols = 4
    n_rows = -(-n_cats // n_cols)  # ceil division, e.g. 8 cats -> 2 rows x 4 cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), sharex=True)
    axes = axes.flatten()

    # Consistent color per run across all 8 subplots.
    run_names = list(runs_eval_data.keys())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    run_colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(run_names)}

    for ax, cat in zip(axes, CATEGORIES):
        for name in run_names:
            eval_rows = runs_eval_data[name]
            epochs = list(range(len(eval_rows)))
            values = [row.get(cat, float("nan")) for row in eval_rows]
            ax.plot(epochs, values, marker="o", markersize=3, label=name, color=run_colors[name])
        ax.set_title(cat)
        ax.grid(alpha=0.3)

    # Hide any unused subplot axes (if n_cats doesn't fill the grid exactly).
    for ax in axes[n_cats:]:
        ax.axis("off")

    for ax in axes[:n_cats]:
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")

    # One shared legend for the whole figure instead of repeating it in every subplot.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(run_names), bbox_to_anchor=(0.5, 1.03))

    fig.suptitle("Per-category accuracy vs. epoch, by spatial pooling config", y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=".", help="where to save the PNGs")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Plot (a): accuracy + loss across all runs ---
    runs_data = {}
    for name, run_dir in RUNS:
        log_path = os.path.join(run_dir, "train_log.txt")
        if not os.path.exists(log_path):
            print(f"[skip] {name}: no train_log.txt found at {log_path}")
            continue
        runs_data[name] = parse_train_log(log_path)

    #plot_accuracy_and_loss(runs_data, os.path.join(args.out_dir, "accuracy_and_loss.png"))

    # --- Plot (b): per-category accuracy, one subplot per category, all runs overlaid ---
    runs_eval_data = {}
    for name, run_dir in RUNS:
        eval_path = os.path.join(run_dir, "evaluate.txt")
        if not os.path.exists(eval_path):
            print(f"[skip] {name}: no evaluate.txt found at {eval_path}")
            continue
        runs_eval_data[name] = parse_evaluate(eval_path)

    plot_categories_grid(runs_eval_data, os.path.join(args.out_dir, "categories_by_config.png"))


if __name__ == "__main__":
    main()