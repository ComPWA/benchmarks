# cspell:disable
from __future__ import annotations

import logging
from collections import defaultdict
from os.path import dirname

import matplotlib.pyplot as plt
import numpy as np
import yaml
from polarization.io import mute_jax_warnings
from polarization.plot import use_mpl_latex_fonts

THIS_DIRECTORY = dirname(__file__)
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.ERROR)
mute_jax_warnings()


def main() -> int:
    imported_times = load_benchmark(f"{THIS_DIRECTORY}/computation_times.yaml")

    x = np.array(sorted(imported_times))
    y = defaultdict(list)
    for n in x:
        dct = imported_times[n]
        for k, v in dct.items():
            y[k].append(v)
    y = {k: np.array(v) for k, v in y.items()}

    use_mpl_latex_fonts()
    plt.rc("font", size=14)
    _, axes = plt.subplots(
        figsize=(10, 5),
        ncols=2,
        tight_layout=True,
    )
    ax1, ax2 = axes
    ax1.set_title("Run 1 (compilation)")
    ax2.set_title("Run 2 (same shape, different data)")
    ax1.set_ylabel("Computation time (s)")
    for ax in axes.flatten():
        ax.set_xlabel("Number of events")
        ax.set_xscale("log")
        ax.grid(axis="y")

    def plot(
        ax,
        key: str,
        label: str,
        x_selector: np.ndarray | None = None,
        logy: bool = False,
    ):
        x_values = x
        y_values = y[key]
        if x_selector is not None:
            x_values = x_values[x_selector]
            y_values = y_values[x_selector, :]
        ax.errorbar(
            x_values,
            y_values.mean(axis=1),
            yerr=y_values.std(axis=1),
            label=label,
            fmt=".",
        )
        if logy:
            ax.set_yscale("log")

    plot(ax1, "parametrized, compilation", "parametrized", x_selector=x <= 1e6)
    plot(ax1, "substituted, compilation", "substituted", x_selector=x <= 1e6)
    plot(ax2, "parametrized, run 1, same shape", "parametrized", logy=True)
    plot(ax2, "substituted, run 1, same shape", "substituted", logy=True)
    axes.flatten()[-1].legend()
    ax1.set_ylim(0, ax1.get_ylim()[1])
    plt.savefig(f"{THIS_DIRECTORY}/computation_times.svg")
    plt.show()

    return 0


def load_benchmark(filename: str) -> dict[str, dict[str, list[float]]]:
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


if "__main__" in __name__:
    raise SystemExit(main())
