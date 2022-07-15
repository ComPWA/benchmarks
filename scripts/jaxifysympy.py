# cspell:disable
from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import partial
from os.path import dirname
from time import time
from typing import TypeVar

import jax
import numpy as np
import polarization.lhcb
import sympy as sp
import yaml
from polarization.amplitude import AmplitudeModel
from polarization.data import create_data_transformer, generate_phasespace_sample
from polarization.io import mute_jax_warnings, perform_cached_doit
from polarization.lhcb import load_model_builder, load_model_parameters
from polarization.lhcb.particle import load_particles
from tensorwaves.function import ParametrizedBackendFunction, PositionalArgumentFunction
from tensorwaves.function._backend import get_backend_modules
from tensorwaves.function.sympy import (
    _sympy_lambdify,
    create_function,
    create_parametrized_function,
)
from tensorwaves.function.sympy._printer import JaxPrinter
from tensorwaves.interface import DataSample, Function
from tqdm.auto import tqdm
from yaml.representer import Representer


def lambdify_to_jax(expr: sp.Expr, use_cse: bool = False):
    symbols = sorted(expr.free_symbols, key=str)
    return _sympy_lambdify(
        expr,
        symbols,
        modules=get_backend_modules("jax"),
        printer=JaxPrinter(),
        use_cse=use_cse,
    )


THIS_DIRECTORY = dirname(__file__)
DATA_DIRECTORY = dirname(polarization.lhcb.__file__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger()
mute_jax_warnings()

yaml.add_representer(defaultdict, Representer.represent_dict)


############################################################
# build the model
############################################################


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


model = create_amplitude_model()

simplified_model = model.full_expression.xreplace(model.parameter_defaults)


############################################################
# lambdification
############################################################

unfoldedexpr = simplified_model.doit()
f1 = lambdify_to_jax(unfoldedexpr)
f2 = jax.jit(f1)

############################################################
# data sample
############################################################

npars = len(unfoldedexpr.free_symbols)
#
warmdata = np.random.uniform(size=(npars, 1000))
rundata = np.random.uniform(size=(npars, 1000))

############################################################
# benchmark
############################################################


def performbenchmark(f, warmdata, rundata):
    start_time = time()
    _ = f(*warmdata).block_until_ready()
    comptime = time() - start_time
    _ = f(*rundata).block_until_ready()
    runtime = (time() - start_time) - comptime
    print(f"comptime: {comptime-runtime:.2g}s runtime: {runtime:.2g}s")


performbenchmark(f1, warmdata, rundata)
# comptime: -0.0041s runtime: 0.037s

performbenchmark(f2, warmdata, rundata)
# comptime: -0.00013s runtime: 0.0048s


############################################################
# using tensor waves
############################################################


preparefunct = create_function(
    unfoldedexpr,
    backend="jax",
)

f3 = preparefunct.function

############################################################
# benchmark
############################################################

performbenchmark(f3, warmdata[0:-1, :], rundata[0:-1, :])
# comptime: 5.1e-05s runtime: 0.00058s
