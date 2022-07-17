# cspell:disable
from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import partial
from os.path import dirname
from time import time
from typing import TypeVar

import attrs
import jax
import polarization.lhcb
import yaml
from polarization.amplitude import AmplitudeModel
from polarization.data import create_data_transformer, generate_phasespace_sample
from polarization.io import mute_jax_warnings, perform_cached_doit
from polarization.lhcb import load_model_builder, load_model_parameters
from polarization.lhcb.particle import load_particles
from tensorwaves.function import ParametrizedBackendFunction, PositionalArgumentFunction
from tensorwaves.function.sympy import create_function, create_parametrized_function
from tensorwaves.interface import DataSample, Function
from tqdm.auto import tqdm
from yaml.representer import Representer

THIS_DIRECTORY = dirname(__file__)
DATA_DIRECTORY = dirname(polarization.lhcb.__file__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger()
mute_jax_warnings()

yaml.add_representer(defaultdict, Representer.represent_dict)

BENCHMARK_CASES = [
    1,
    3,
    10,
    30,
    100,
    300,
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    10_000_000,
    30_000_000,
]
NUMBER_OF_RUNS = 5


def main() -> int:
    t: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    output_file = f"{THIS_DIRECTORY}/computation_times.yaml"
    if os.path.exists(output_file):
        imported_times = load_benchmark(output_file)
        t.update(imported_times)

    model = create_amplitude_model()
    parametrized_func, substituted_func = prepare_functions(model)

    progress_bar = tqdm(
        desc="Benchmarking intensity evaluation with JAX",
        total=NUMBER_OF_RUNS * len(BENCHMARK_CASES),
    )
    for n in BENCHMARK_CASES:
        progress_bar.set_postfix_str(f"{n:,}-event sample")
        existing_benchmark = t.get(n)
        if existing_benchmark is not None:
            if all(len(v) == NUMBER_OF_RUNS for v in existing_benchmark.values()):
                logging.warning(f"Benchmark for {n:,} events already exists")
                progress_bar.update(NUMBER_OF_RUNS)
                continue
            else:
                t[n] = defaultdict(list)
        warmup_sample = generate_sample(model, n, seed=123456)
        run_sample = generate_sample(model, n, seed=0)
        for _ in range(NUMBER_OF_RUNS):
            func = recompile_jax_function(parametrized_func)
            t[n]["parametrized, compilation"].append(benchmark(func, warmup_sample))
            t[n]["parametrized, run 1, same shape"].append(benchmark(func, run_sample))
            t[n]["parametrized, run 2, same data"].append(benchmark(func, run_sample))

            func = recompile_jax_function(substituted_func)
            t[n]["substituted, compilation"].append(benchmark(func, warmup_sample))
            t[n]["substituted, run 1, same shape"].append(benchmark(func, run_sample))
            t[n]["substituted, run 2, same data"].append(benchmark(func, run_sample))
            progress_bar.update()
        write_benchmark(t, output_file)
    progress_bar.close()

    return 0


def create_amplitude_model() -> AmplitudeModel:
    model_choice = 0
    model_file = f"{DATA_DIRECTORY}/model-definitions.yaml"
    particles = load_particles(f"{DATA_DIRECTORY}/particle-definitions.yaml")
    amplitude_builder = load_model_builder(model_file, particles, model_choice)
    imported_parameter_values = load_model_parameters(
        model_file, amplitude_builder.decay, model_choice
    )
    reference_subsystem = 1
    model = amplitude_builder.formulate(reference_subsystem)
    model.parameter_defaults.update(imported_parameter_values)
    return model


def prepare_functions(
    model: AmplitudeModel,
) -> tuple[ParametrizedBackendFunction, PositionalArgumentFunction]:
    original_log_level = LOGGER.getEffectiveLevel()
    LOGGER.setLevel(logging.INFO)
    logging.info("Unfolding intensity expression")
    unfolded_intensity_expr = perform_cached_doit(model.full_expression)
    logging.info("Substituting parameters")
    substituted_expr = unfolded_intensity_expr.xreplace(model.parameter_defaults)
    logging.info("Lambdifying full intensity expression")
    parametrized_func = create_parametrized_function(
        unfolded_intensity_expr,
        parameters=model.parameter_defaults,
        backend="jax",
    )
    logging.info("Lambdifying substituted intensity expression")
    substituted_func = create_function(substituted_expr, backend="jax")
    logging.info("Finished function lambdification")
    LOGGER.setLevel(original_log_level)
    return parametrized_func, substituted_func


def generate_sample(
    model: AmplitudeModel, n_events: int, seed: int | None = None
) -> DataSample:
    transformer = create_data_transformer(model)
    original_log_level = LOGGER.getEffectiveLevel()
    LOGGER.setLevel(logging.ERROR)
    phsp_sample = generate_phasespace_sample(model.decay, n_events, seed)
    LOGGER.setLevel(original_log_level)
    phsp_sample = transformer(phsp_sample)
    return phsp_sample


T = TypeVar("T", ParametrizedBackendFunction, PositionalArgumentFunction)


def recompile_jax_function(func: T) -> T:
    def recompile(f):
        # https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html#caching
        return jax.jit(partial(f))

    if isinstance(func, ParametrizedBackendFunction):
        return ParametrizedBackendFunction(
            function=recompile(func.function),
            argument_order=func.argument_order,
            parameters=func.parameters,
        )
    return attrs.evolve(func, function=recompile(func.function))


def benchmark(func: Function, sample: DataSample) -> float:
    # https://jax.rtfd.io/en/latest/async_dispatch.html
    start = time()
    func(sample).block_until_ready()
    stop = time()
    return stop - start


def load_benchmark(filename: str) -> dict[str, dict[str, list[float]]]:
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def write_benchmark(times: dict[str, dict[str, list[float]]], filename: str) -> None:
    with open(filename, "w") as f:
        yaml.dump(
            times,
            f,
            default_flow_style=False,
            sort_keys=True,
            Dumper=IncreasedIndent,
        )


class IncreasedIndent(yaml.Dumper):
    # pylint: disable=too-many-ancestors
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)

    def write_line_break(self, data: str | None = None) -> None:
        """See https://stackoverflow.com/a/44284819."""
        super().write_line_break(data)
        if len(self.indents) == 1:
            super().write_line_break()


if "__main__" in __name__:
    raise SystemExit(main())
