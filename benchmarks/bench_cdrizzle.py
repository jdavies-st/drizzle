"""
Timing benchmarks for the C drizzle and blot code.

Run from the repository root with the extension built in place::

    python benchmarks/bench_cdrizzle.py

To compare two builds, keep a copy of the compiled extension from one of them
(for example, built from main) and pass it with ``--against``::

    cp drizzle/cdrizzle*.so /tmp/cdrizzle_main.so
    # ... switch branch and rebuild ...
    python benchmarks/bench_cdrizzle.py --against /tmp/cdrizzle_main.so

Both builds are then loaded into the same process and timed alternately,
which cancels most of the drift in CPU clock speed and core scheduling that
makes separate runs hard to compare, especially on laptops. Results can also
be saved to and compared with JSON files (``--save`` and ``--compare``) to
track timings on a quiet machine.

Each case is timed ``--repeat`` times after one untimed run. Ratios use the
minimum time, which is the least sensitive to other load on the machine.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import platform
import statistics
import time
import warnings

import numpy as np

import drizzle
from drizzle import cdrizzle

DRIZZLE_KERNELS = ["square", "turbo", "point", "gaussian", "lanczos2", "lanczos3"]
BLOT_INTERPOLATORS = ["nearest", "linear", "poly3", "poly5", "lanczos3", "lanczos5"]


def load_extension(path):
    """Load a compiled cdrizzle extension from any file path."""
    loader = importlib.machinery.ExtensionFileLoader("cdrizzle", path)
    spec = importlib.util.spec_from_loader("cdrizzle", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def make_inputs(size):
    """Detector-sized image with a rotated and slightly distorted pixel map."""
    rng = np.random.default_rng(12345)
    data = rng.random((size, size), dtype=np.float32) + 1.0
    y, x = np.indices((size, size), dtype=np.float64)
    xc = x - size / 2
    yc = y - size / 2
    xo = size / 2 + 0.96 * xc - 0.28 * yc + 1.0e-5 * xc * xc
    yo = size / 2 + 0.28 * xc + 0.96 * yc + 1.0e-5 * xc * yc
    return data, np.dstack([xo, yo])


def drizzle_case(kernel, data, pixmap, nvar=0):
    weights = np.ones_like(data)
    kwargs = {}
    if nvar:
        kwargs["input2"] = [data] * nvar
        kwargs["dq"] = np.zeros(data.shape, dtype=np.uint32)

    def run(mod):
        # np.full writes every element, so that page faults on first touch
        # of new output arrays are not timed; Drizzle reuses its outputs.
        if nvar:
            kwargs["output2"] = [np.full_like(data, 0.0) for _ in range(nvar)]
            kwargs["outdq"] = np.full(data.shape, 0, dtype=np.uint32)
        out = np.full_like(data, 0.0)
        counts = np.full_like(data, 0.0)
        context = np.full(data.shape, 0, dtype=np.int32)
        t0 = time.perf_counter()
        mod.tdriz(
            data,
            weights,
            pixmap,
            out,
            counts,
            context,
            uniqid=1,
            pixfrac=1.0,
            kernel=kernel,
            in_units="cps",
            expscale=1.0,
            wtscale=1.0,
            fillstr="INDEF",
            **kwargs,
        )
        return time.perf_counter() - t0

    return run


def blot_case(interp, data, pixmap):
    def run(mod):
        out = np.full_like(data, 0.0)
        t0 = time.perf_counter()
        mod.tblot(data, pixmap, out, interp=interp)
        return time.perf_counter() - t0

    return run


def make_cases(size):
    data, pixmap = make_inputs(size)
    cases = {}
    for kernel in DRIZZLE_KERNELS:
        cases[f"tdriz_{kernel}"] = drizzle_case(kernel, data, pixmap)
    # the way jwst resamples: variance arrays and DQ alongside the science data
    cases["tdriz_square_var3_dq"] = drizzle_case("square", data, pixmap, nvar=3)
    for interp in BLOT_INTERPOLATORS:
        cases[f"tblot_{interp}"] = blot_case(interp, data, pixmap)
    return cases


def time_case(run, modules, repeat):
    """Time a case with each module, alternating the order on each repeat."""
    for mod in modules.values():
        run(mod)
    times = {name: [] for name in modules}
    order = list(modules)
    for _ in range(repeat):
        for name in order:
            times[name].append(run(modules[name]))
        order.reverse()
    return {name: {"min": min(t), "median": statistics.median(t)} for name, t in times.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--size", type=int, default=2048, help="image size in pixels")
    parser.add_argument("--repeat", type=int, default=10, help="timings per case")
    parser.add_argument("-k", dest="select", help="only run cases containing this string")
    parser.add_argument("--against", metavar="SO", help="compiled extension to compare with")
    parser.add_argument("--save", metavar="FILE", help="write results to a JSON file")
    parser.add_argument("--compare", metavar="FILE", help="compare with saved JSON results")
    args = parser.parse_args()

    modules = {"this": cdrizzle}
    if args.against:
        modules["other"] = load_extension(args.against)

    baseline = None
    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        if baseline["meta"]["size"] != args.size:
            parser.error(f"baseline was run with --size {baseline['meta']['size']}")

    cases = make_cases(args.size)
    if args.select:
        cases = {name: case for name, case in cases.items() if args.select in name}

    header = f"{'case':24s} {'min (s)':>9s} {'median (s)':>11s}"
    if args.against:
        header += f" {'other min':>10s} {'other med':>10s} {'ratio':>7s}"
    if baseline:
        header += f" {'base min':>9s} {'ratio':>7s}"
    print(header)

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, run in cases.items():
            timings = time_case(run, modules, args.repeat)
            this = timings["this"]
            results[name] = this
            line = f"{name:24s} {this['min']:9.4f} {this['median']:11.4f}"
            if args.against:
                other = timings["other"]
                ratio = this["min"] / other["min"]
                line += f" {other['min']:10.4f} {other['median']:10.4f} {ratio:7.3f}"
            if baseline and name in baseline["results"]:
                base = baseline["results"][name]["min"]
                line += f" {base:9.4f} {this['min'] / base:7.3f}"
            print(line, flush=True)

    if args.save:
        meta = {
            "size": args.size,
            "repeat": args.repeat,
            "drizzle": drizzle.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
        }
        with open(args.save, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"Wrote {args.save}")


if __name__ == "__main__":
    main()
