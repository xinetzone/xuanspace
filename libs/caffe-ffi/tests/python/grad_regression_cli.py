#!/usr/bin/env python3
"""Gradient regression CLI for CI pipelines.

Runs C++ backward vs numpy reference backward (and optionally numerical
gradient cross-check) for a caffe-ffi Net, producing structured JSON output
and proper exit codes for CI integration.

Exit codes:
    0  PASS  — all checks passed
    1  FAIL  — gradient mismatch detected
    2  ERROR — usage error / import failure / network construction error

Usage examples:
    # Quick numerical-only check (no reference needed)
    python grad_regression_cli.py --prototxt net.prototxt \\
        --input data --output pool --dims 1 1 4 4 --skip-numerical=false

    # Reference + numerical check for pooling
    python grad_regression_cli.py --prototxt net.prototxt \\
        --input data --output pool --dims 1 2 5 5 --seed 42 \\
        --ref tests.python.test_pooling_backward:pooling_backward_np \\
        --kwarg kernel_size=2 --kwarg stride=2 --kwarg pool_type=MAX

    # JSON output for CI artifact
    python grad_regression_cli.py --prototxt net.prototxt \\
        --input data --output pool --dims 1 1 4 4 --json result.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Ensure tests/python is importable when run as a script
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
# Ensure caffe_ffi is importable (project root may need to be on path)
# _THIS_DIR = tests/python/ → .parent = tests/ → .parent.parent = caffe-ffi/ (project root)
_PROJECT_ROOT = _THIS_DIR.parent.parent  # libs/caffe-ffi
# Add both project root (for 'tests.python.*' and 'caffe_ffi' imports) and
# tests/python (for direct sibling imports like _grad_check_utils)
for candidate in [_PROJECT_ROOT, _THIS_DIR.parent, _THIS_DIR]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from _grad_check_utils import (  # noqa: E402
    assert_backward_matches_reference,
    compare_gradients,
    numerical_grad_for_input,
)

LOG = logging.getLogger("grad_regression_cli")


# ---------------------------------------------------------------------------
# Prototxt loading
# ---------------------------------------------------------------------------

def _load_prototxt(source: str) -> str:
    """Load prototxt from file path or stdin marker ('-')."""
    if source == "-":
        return sys.stdin.read()
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Prototxt file not found: {source}")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Reference function loader
# ---------------------------------------------------------------------------

def _load_ref_fn(ref_spec: str) -> Callable:
    """Load a reference backward function from 'module.path:function_name'."""
    if ":" not in ref_spec:
        raise ValueError(
            f"--ref must be 'module.path:function_name', got: {ref_spec!r}"
        )
    module_path, fn_name = ref_spec.split(":", 1)
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Cannot import module {module_path!r}: {e}") from e
    if not hasattr(mod, fn_name):
        raise AttributeError(
            f"Module {module_path!r} has no attribute {fn_name!r}"
        )
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"{module_path}.{fn_name} is not callable")
    return fn


# ---------------------------------------------------------------------------
# Kwarg parsing: --kwarg key=value (auto-coerce int/float/bool/str)
# ---------------------------------------------------------------------------

def _parse_kwargs(kwarg_list: list[str]) -> dict[str, Any]:
    """Parse list of 'key=value' strings into a dict with type coercion."""
    result: dict[str, Any] = {}
    for item in kwarg_list:
        if "=" not in item:
            raise ValueError(f"--kwarg must be key=value, got: {item!r}")
        key, raw_val = item.split("=", 1)
        result[key.strip()] = _coerce_value(raw_val.strip())
    return result


def _coerce_value(raw: str) -> Any:
    """Auto-coerce string value to int, float, bool, or keep as string."""
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    if raw.lower() in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


# ---------------------------------------------------------------------------
# Net construction
# ---------------------------------------------------------------------------

def _build_net(prototxt: str):
    """Build a caffe_ffi Net from prototxt string."""
    try:
        from caffe_ffi import Net
    except ImportError as e:
        raise ImportError(
            f"Cannot import caffe_ffi: {e}. "
            "Ensure C++ extension is built and PYTHONPATH includes build dir."
        ) from e
    return Net(prototxt)


# ---------------------------------------------------------------------------
# Main regression logic
# ---------------------------------------------------------------------------

def run_regression(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the regression test and return a result dict."""
    result: dict[str, Any] = {
        "status": "ERROR",
        "input": args.input,
        "output": args.output,
        "dims": args.dims,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Load prototxt
    try:
        prototxt = _load_prototxt(args.prototxt)
    except (FileNotFoundError, OSError) as e:
        result["error"] = str(e)
        result["status"] = "ERROR"
        return result
    result["prototxt_source"] = args.prototxt

    # Build net
    try:
        net = _build_net(prototxt)
    except Exception as e:
        result["error"] = f"Failed to build Net: {e}"
        result["status"] = "ERROR"
        return result

    # Load reference function if specified
    ref_fn = None
    if args.ref:
        try:
            ref_fn = _load_ref_fn(args.ref)
            result["ref_fn"] = args.ref
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            result["error"] = f"Failed to load reference function: {e}"
            result["status"] = "ERROR"
            return result

    # Parse extra kwargs
    try:
        ref_kwargs = _parse_kwargs(args.kwarg)
    except ValueError as e:
        result["error"] = str(e)
        result["status"] = "ERROR"
        return result
    result["ref_kwargs"] = ref_kwargs

    # Generate input
    rng = np.random.RandomState(args.seed)
    x = rng.randn(*args.dims).astype(np.float32)
    # Get output shape from a forward pass (dry-run) to determine dy shape
    try:
        dry_out = net.forward({args.input: x})
        if args.output not in dry_out:
            available = list(dry_out.keys())
            result["error"] = (
                f"Output blob {args.output!r} not found. "
                f"Available outputs: {available}"
            )
            result["status"] = "ERROR"
            return result
        out_shape = dry_out[args.output].shape
    except Exception as e:
        result["error"] = f"Forward pass failed: {e}"
        result["status"] = "ERROR"
        return result

    dy = rng.randn(*out_shape).astype(np.float32) * 0.1
    result["out_shape"] = list(out_shape)

    start = time.perf_counter()

    if ref_fn is not None:
        # Use assert_backward_matches_reference for full regression check
        try:
            check_result = assert_backward_matches_reference(
                net,
                ref_fn,
                input_name=args.input,
                output_name=args.output,
                x=x,
                dy=dy,
                name=args.name or f"{args.input}->{args.output}",
                rtol=args.rtol,
                atol=args.atol,
                verbose=not args.quiet,
                skip_numerical=args.skip_numerical,
                numerical_h=args.numerical_h,
                numerical_rtol=args.numerical_rtol,
                numerical_atol=args.numerical_atol,
                **ref_kwargs,
            )
            result["ref_passed"] = check_result["ref_passed"]
            result["numerical_passed"] = check_result["numerical_passed"]
            result["status"] = "PASS" if (
                check_result["ref_passed"]
                and (check_result["numerical_passed"] is None
                     or check_result["numerical_passed"])
            ) else "FAIL"
            if check_result.get("ref_info"):
                info = check_result["ref_info"]
                result["ref_max_err"] = float(info.get("max_abs_err", 0))
                result["ref_max_rel"] = float(info.get("max_rel_err", 0))
            if check_result.get("numerical_info"):
                ninfo = check_result["numerical_info"]
                result["num_max_err"] = float(ninfo.get("max_abs_err", 0))
                result["num_max_rel"] = float(ninfo.get("max_rel_err", 0))
        except AssertionError as e:
            result["status"] = "FAIL"
            result["error"] = str(e)
        except Exception as e:
            result["status"] = "ERROR"
            result["error"] = f"Check failed with exception: {e}"
    else:
        # No reference: numerical gradient check only
        try:
            # Need to rebuild net because dry-run forward consumed it
            net2 = _build_net(prototxt)
            net2.forward({args.input: x})
            net2.backward({args.output: dy})
            analytic_dX = net2.blob_by_name(args.input).diff

            numerical_dX = numerical_grad_for_input(
                net2, args.input, x, args.output, dy,
                h=args.numerical_h,
                name=args.name or f"{args.input}->{args.output}",
                verbose=not args.quiet,
            )
            cmp = compare_gradients(
                analytic_dX, numerical_dX,
                name=args.name or f"{args.input}->{args.output}",
                rtol=args.numerical_rtol,
                atol=args.numerical_atol,
                verbose=not args.quiet,
            )
            result["numerical_passed"] = cmp["passed"]
            result["num_max_err"] = float(cmp.get("max_abs_err", 0))
            result["num_max_rel"] = float(cmp.get("max_rel_err", 0))
            result["ref_passed"] = None
            result["status"] = "PASS" if cmp["passed"] else "FAIL"
        except AssertionError as e:
            result["status"] = "FAIL"
            result["error"] = str(e)
        except Exception as e:
            result["status"] = "ERROR"
            result["error"] = f"Numerical check failed: {e}"

    elapsed = time.perf_counter() - start
    result["elapsed_ms"] = round(elapsed * 1000, 1)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="C++ backward gradient regression CLI for CI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--prototxt", required=True,
                   help="Path to prototxt file, or '-' for stdin")
    p.add_argument("--input", required=True, help="Input blob name")
    p.add_argument("--output", required=True, help="Output blob name")
    p.add_argument("--dims", type=int, nargs=4, required=True, metavar=("N","C","H","W"),
                   help="Input tensor dimensions (N C H W)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--name", default=None, help="Label for log messages")

    # Reference function
    p.add_argument("--ref", default=None,
                   help="Numpy reference fn as 'module.path:function_name'")
    p.add_argument("--kwarg", action="append", default=[],
                   help="Extra keyword arg for ref fn, as key=value (repeatable)")

    # Tolerances for ref comparison
    p.add_argument("--rtol", type=float, default=1e-3,
                   help="Relative tolerance for ref vs C++ (default: 1e-3)")
    p.add_argument("--atol", type=float, default=1e-4,
                   help="Absolute tolerance for ref vs C++ (default: 1e-4)")

    # Numerical gradient options
    p.add_argument("--skip-numerical", action="store_true",
                   help="Skip numerical gradient cross-check")
    p.add_argument("--numerical-h", type=float, default=1e-3,
                   help="Finite-difference step size (default: 1e-3)")
    p.add_argument("--numerical-rtol", type=float, default=1e-2,
                   help="Relative tolerance for numerical check (default: 1e-2)")
    p.add_argument("--numerical-atol", type=float, default=1e-3,
                   help="Absolute tolerance for numerical check (default: 1e-3)")

    # Output
    p.add_argument("--json", default=None,
                   help="Write JSON result to file (use '-' for stdout)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress verbose gradient logs")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    result = run_regression(args)

    # JSON output
    if args.json is not None:
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        if args.json == "-":
            print(json_str)
        else:
            Path(args.json).write_text(json_str + "\n", encoding="utf-8")
            if not args.quiet:
                print(f"[CLI] JSON result written to {args.json}", file=sys.stderr)

    status = result["status"]
    if status == "PASS":
        print(f"[CLI] PASS — {result.get('ref_passed')=} {result.get('numerical_passed')=} "
              f"({result.get('elapsed_ms', 0)}ms)", file=sys.stderr)
        return 0
    elif status == "FAIL":
        print(f"[CLI] FAIL — {result.get('error', 'gradient mismatch')}", file=sys.stderr)
        return 1
    else:
        print(f"[CLI] ERROR — {result.get('error', 'unknown error')}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
