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
    fig, axes = plt.subplots(figsize=(10, 5), ncols=2, tight_layout=True)
    ax1, ax2 = axes
    fig.suptitle("JAX performance")
    ax1.set_title("First run")
    ax2.set_title("Second run (XLA cache)")
    ax1.set_ylabel("Computation time (s)")
    ax2.set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Number of events")
        ax.set_xscale("log")
        ax.grid(axis="y")
    style = dict(
        fmt=".",
    )

    ax1.errorbar(
        x,
        y["parametrized, run 1"].mean(axis=1),
        yerr=y["parametrized, run 1"].std(axis=1),
        label="parametrized",
        **style,
    )
    ax1.errorbar(
        x,
        y["substituted, run 1"].mean(axis=1),
        yerr=y["substituted, run 1"].std(axis=1),
        label="substituted",
        **style,
    )

    ax2.errorbar(
        x,
        y["parametrized, run 2"].mean(axis=1),
        yerr=y["parametrized, run 2"].std(axis=1),
        label="parametrized",
        **style,
    )
    ax2.errorbar(
        x,
        y["substituted, run 2"].mean(axis=1),
        yerr=y["substituted, run 2"].std(axis=1),
        label="substituted",
        **style,
    )
    ax1.legend()
    ax1.set_ylim(0, ax1.get_ylim()[1])
    plt.savefig(f"{THIS_DIRECTORY}/computation_times.svg")
    plt.show()

    return 0


def load_benchmark(filename: str) -> dict[str, dict[str, list[float]]]:
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


if "__main__" in __name__:
    raise SystemExit(main())
