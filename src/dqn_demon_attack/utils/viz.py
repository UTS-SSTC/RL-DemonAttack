"""
Training log visualization utilities.

Reads CSV log files and generates plots for key training metrics including
episode returns, episode length, loss, Q-values, and exploration epsilon.
"""

import argparse
import csv

import matplotlib.pyplot as plt


def moving_average(xs, k=20):
    """
    Compute moving average over a sequence.

    Args:
        xs: Input sequence of values.
        k: Window size for moving average.

    Returns:
        List of moving averaged values with same length as input.
    """
    if k <= 1:
        return xs

    out = []
    s = 0.0
    q = []

    for x in xs:
        q.append(x)
        s += x
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))

    return out


def read_csv(path):
    """
    Read CSV file into list of dictionaries.

    Args:
        path: Path to CSV file.

    Returns:
        List of row dictionaries.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def to_float(xs, key):
    """
    Extract float values from CSV rows.

    Args:
        xs: List of row dictionaries.
        key: Column key to extract.

    Returns:
        List of float values, with NaN for invalid entries.
    """
    out = []
    for r in xs:
        try:
            out.append(float(r[key]))
        except Exception:
            out.append(float("nan"))
    return out


def plot_series(x, y, title, xlabel, ylabel, smooth=50):
    """
    Plot a time series with optional smoothing.

    Args:
        x: X-axis values.
        y: Y-axis values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        smooth: Window size for moving average smoothing. Set to 1 or None to disable.
    """
    plt.figure()

    if smooth and smooth > 1:
        y_s = moving_average(y, k=smooth)
        plt.plot(x, y_s, label=f"{ylabel} (MA{smooth})")

    plt.plot(x, y, alpha=0.3, label=ylabel)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main(log_path):
    """
    Generate training metric plots from CSV log.

    Args:
        log_path: Path to training log CSV file.
    """
    rows = read_csv(log_path)
    step = to_float(rows, "step")
    ep_return_raw = to_float(rows, "ep_return_raw")
    ep_return_shaped = to_float(rows, "ep_return_shaped")
    ep_len = to_float(rows, "ep_len")
    loss = to_float(rows, "loss")
    q_mean = to_float(rows, "q_mean")
    eps = to_float(rows, "epsilon")

    plot_series(step, ep_return_raw, "Episode Raw Return vs Step", "Step", "Raw Return")
    plot_series(step, ep_return_shaped, "Episode Shaped Return vs Step", "Step", "Shaped Return")
    plot_series(step, ep_len, "Episode Length vs Step", "Step", "Episode Length")
    plot_series(step, loss, "Loss vs Step", "Step", "Loss")
    plot_series(step, q_mean, "Mean Q vs Step", "Step", "Mean Q")
    plot_series(step, eps, "Epsilon vs Step", "Step", "Epsilon", smooth=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training logs")
    parser.add_argument("--log", type=str, default="runs/exp1/train_log.csv", help="Path to training log CSV")
    args = parser.parse_args()
    main(args.log)
