#!/usr/bin/env python3
"""
Numerical testing safety checker for activation function tests.

Checks two categories of issues in Python test files:

1. C¹ kink protection (C¹拐点防护):
   Numerical gradient tests for C¹-discontinuous activation functions
   (LeakyReLU, PReLU, ELU α≠1) must push test points away from the kink
   at x=0 using `avoid_c1_discontinuity()`.

2. ULP saturation violations (ULP饱和违规):
   Assertions that demand precision tighter than float32 ULP in saturation
   regions are impossible to satisfy. E.g. `sigmoid(80) > 1.0 - 1e-10`
   fails because sigmoid(80) rounds exactly to 1.0 in float32, so
   1.0 > 1.0 - 1e-10 = 0.9999999999 is true but the implication is wrong;
   the real issue is `assert sigmoid(80) != 1.0` or asserting sub-ULP gaps.

Exit codes:
    0 - all checks pass (or no relevant tests found)
    1 - violations found
    2 - usage error (invalid arguments, path not found)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ── Pattern definitions: C¹ kink ──────────────────────────────────────

# Patterns indicating a C¹-discontinuous activation layer is being tested.
# Standard ReLU (negative_slope=0) uses offset strategy and is only warned about,
# not failed (because with fixed-seed random offset +1.0 it's probabilistically safe).
C1_DISCONTINUOUS_PATTERNS: list[tuple[str, re.Pattern, bool]] = [
    # PReLU layer type (always C¹-discontinuous regardless of parameters)
    (
        "PReLU",
        re.compile(r"""type\s*:\s*["']PReLU["']""", re.IGNORECASE),
        True,  # fail-level violation if unprotected
    ),
    # LeakyReLU via negative_slope > 0 (ReLU with negative_slope parameter)
    (
        "LeakyReLU(negative_slope>0)",
        re.compile(
            r"""negative_slope\s*[=:]\s*(?!0+(?:\.0+)?(?![.\d]))(\d+\.?\d*|\.\d+)""",
            re.IGNORECASE,
        ),
        True,  # fail-level violation if unprotected
    ),
    # Generic ReLU with non-zero negative_slope (alternative param format)
    (
        "ReLU(negative_slope>0)",
        re.compile(
            r"""(?:relu_param|leaky_relu_param)\s*\{[^}]*negative_slope\s*:\s*(?!0+(?:\.0+)?(?![.\d]))(\d+\.?\d*|\.\d+)""",
            re.IGNORECASE | re.DOTALL,
        ),
        True,
    ),
    # ELU with alpha ≠ 1.0 (C¹-discontinuous at x=0 because f'(0⁻)=α≠1=f'(0⁺))
    # Detects two formats:
    #   - prototxt: elu_param { alpha: <not-1> }
    #   - Python kwargs: any_func(..., alpha=<not-1>, ...)
    (
        "ELU(alpha≠1)",
        re.compile(
            r"""(?:elu_param\s*\{[^}]*alpha\s*:\s*(?!1(?:\.0+)?(?![.\d]))(\d+\.?\d*|\.\d+)"""
            r"""|[a-zA-Z_]\w*\s*\([^)]*\balpha\s*=\s*(?!1(?:\.0+)?(?![.\d]))(\d+\.?\d*|\.\d+))""",
            re.IGNORECASE | re.DOTALL,
        ),
        True,
    ),
]

# Standard ReLU (no negative_slope, or negative_slope=0) — warning only
STANDARD_RELU_PATTERN = re.compile(
    r"""type\s*:\s*["']ReLU["']""",
    re.IGNORECASE,
)

# Patterns indicating numerical gradient checking
NUMERICAL_GRAD_PATTERNS: list[re.Pattern] = [
    re.compile(r"_num_grad\s*\(", re.IGNORECASE),
    re.compile(r"(?:lp|l_plus)\s*-\s*(?:lm|l_minus)\s*\)?\s*/\s*\(?\s*2\s*\*?\s*h", re.IGNORECASE),
    re.compile(r"(?:lp|l_plus)\s*-\s*(?:lm|l_minus)\s*\)?\s*/\s*\(?\s*\(?\s*2\s*[*]?\s*h?\)?", re.IGNORECASE),
    re.compile(r"numerical.*grad|grad.*numerical|finite.?diff", re.IGNORECASE),
]

# Pattern indicating avoid_c1_discontinuity is used
AVOID_CALL_PATTERN = re.compile(r"avoid_c1_discontinuity\s*\(", re.IGNORECASE)

# Suppression comment patterns (per-file or per-test opt-out with justification)
C1_SUPPRESSION_PATTERNS: list[re.Pattern] = [
    re.compile(r"#\s*c1-kink-ok\b", re.IGNORECASE),
    re.compile(r"#\s*c1-kink:\s*(?:intentional|exempt|offset|allow|special)\b", re.IGNORECASE),
    re.compile(r"#\s*intentionally.*(?:cross|straddle|span|hit)\s*(?:the\s*)?kink", re.IGNORECASE),
]

# Filename patterns that are automatically exempt (specialized kink validation tests)
EXEMPT_FILENAME_PATTERNS: list[re.Pattern] = [
    re.compile(r"kink.*stability", re.IGNORECASE),
    re.compile(r"c1.*kink.*test", re.IGNORECASE),
    re.compile(r"test_elu_kink", re.IGNORECASE),
]

# Helper module import patterns
HELPER_IMPORT_PATTERNS: list[re.Pattern] = [
    re.compile(r"from\s+.*caffe_test_helpers\s+import\s+.*avoid_c1_discontinuity", re.IGNORECASE),
    re.compile(r"import\s+.*caffe_test_helpers", re.IGNORECASE),
]


# ── Pattern definitions: ULP saturation ────────────────────────────────

# float32 ULP bounds (verified experimentally, 2026-08-02):
#   ULP(1.0)         ≈ 1.19e-7
#   half-ULP(1.0)    ≈ 6.0e-8   (rounding threshold)
#   FLT_MIN (normal) ≈ 1.18e-38
#   FLT_TRUE_MIN     ≈ 1.40e-45 (subnormal)
#
# Any assertion demanding tighter precision than half-ULP on saturation
# values is mathematically impossible in float32.

# Pattern: `> 1.0 - 1e-N` or `>= 1 - 1e-N` where N >= 8 (epsilon <= 1e-8 < half-ULP ≈ 6e-8)
# Catches: sigmoid(80) > 1 - 1e-10, tanh(100) >= 1.0 - 1e-15, etc.
ULP_TIGHT_GAP_PATTERN = re.compile(
    r"""(?:[><])=?\s*1(?:\.0+)?\s*-\s*1e-0*([1-9]\d|[89])(?!\d)""",
    re.IGNORECASE,
)
# Warning for borderline: epsilon = 1e-7 (≈ ULP, marginal)
ULP_BORDERLINE_GAP_PATTERN = re.compile(
    r"""(?:[><])=?\s*1(?:\.0+)?\s*-\s*1e-0*7(?!\d)""",
    re.IGNORECASE,
)

# Pattern: asserting `< float32_subnormal` (e.g., < 1e-46) which is impossible
ULP_IMPOSSIBLY_SMALL_PATTERN = re.compile(
    r"""<\s*1e-0*(?:4[6-9]|[5-9]\d)\b""",
    re.IGNORECASE,
)

# Saturation function references (used to narrow ULP checks to relevant assertions)
# (?<![a-zA-Z0-9]) allows underscore prefixes (e.g. _sigmoid, custom_tanh)
SAT_FN_PATTERN = re.compile(
    r"""(?<![a-zA-Z0-9])(?:sigmoid|tanh|softmax|exp)\s*\(""",
    re.IGNORECASE,
)

# ULP suppression comment
ULP_SUPPRESSION_PATTERNS: list[re.Pattern] = [
    re.compile(r"#\s*ulp-ok\b", re.IGNORECASE),
    re.compile(r"#\s*ulp:\s*(?:intentional|exempt|verified|precise|analytic)\b", re.IGNORECASE),
    re.compile(r"#\s*intentionally.*(?:ulp|saturation|precise)", re.IGNORECASE),
]


# ── Terminal colors ────────────────────────────────────────────────────


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def enabled(cls) -> bool:
        import os

        if os.environ.get("NO_COLOR"):
            return False
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def c(text: str, code: str) -> str:
    return f"{code}{text}{Colors.RESET}" if Colors.enabled() else text


# ── Core checking logic: C¹ kink ──────────────────────────────────────


def is_exempt_filename(filepath: Path) -> bool:
    """Check if a filename matches exempt patterns (kink validation tests)."""
    name = filepath.name
    return any(p.search(name) for p in EXEMPT_FILENAME_PATTERNS)


def has_c1_suppression(content: str) -> bool:
    """Check if the file has a file-level C¹ kink suppression comment."""
    lines = content.splitlines()[:20]
    header = "\n".join(lines)
    return any(p.search(header) for p in C1_SUPPRESSION_PATTERNS)


def has_ulp_suppression(content: str, line_idx: int, lines: list[str]) -> bool:
    """Check if a specific line has a ULP suppression comment (inline or line above)."""
    # Check inline comment on same line
    line = lines[line_idx]
    if any(p.search(line) for p in ULP_SUPPRESSION_PATTERNS):
        return True
    # Check comment on the line immediately above
    if line_idx > 0:
        prev = lines[line_idx - 1]
        if any(p.search(prev) for p in ULP_SUPPRESSION_PATTERNS):
            return True
    return False


def find_c1_discontinuous_layers(content: str) -> list[tuple[str, int]]:
    """Find all C¹-discontinuous layer references in the file.

    Returns list of (layer_type_name, line_number) tuples.
    """
    findings: list[tuple[str, int]] = []
    seen_lines: set[int] = set()

    for layer_name, pattern, _is_critical in C1_DISCONTINUOUS_PATTERNS:
        for match in pattern.finditer(content):
            line_no = content[: match.start()].count("\n") + 1
            if line_no not in seen_lines:
                findings.append((layer_name, line_no))
                seen_lines.add(line_no)

    return findings


def has_numerical_gradient_check(content: str) -> list[int]:
    """Check if the file contains numerical gradient checking code.

    Returns list of line numbers where numerical gradient patterns are found.
    """
    lines: list[int] = []
    seen: set[int] = set()
    for pattern in NUMERICAL_GRAD_PATTERNS:
        for match in pattern.finditer(content):
            line_no = content[: match.start()].count("\n") + 1
            if line_no not in seen:
                lines.append(line_no)
                seen.add(line_no)
    return lines


def has_kink_protection(content: str) -> bool:
    """Check if the file uses avoid_c1_discontinuity or helper import."""
    if AVOID_CALL_PATTERN.search(content):
        return True
    return any(p.search(content) for p in HELPER_IMPORT_PATTERNS)


# ── Core checking logic: ULP saturation ───────────────────────────────


def check_ulp_saturation(content: str, filepath: Path) -> tuple[list[str], list[str]]:
    """Check for ULP saturation assertion violations.

    Detects assertions that demand precision tighter than float32 ULP
    on saturated activation outputs. Returns (errors, warnings).
    """
    errors: list[str] = []
    warnings: list[str] = []
    lines = content.splitlines()

    # Pre-filter: files that don't reference saturation functions
    # still need checking (e.g., direct assertions on outputs), but we
    # skip files with no assert statements for performance
    has_assert = any("assert" in line for line in lines)
    if not has_assert:
        return errors, warnings

    for i, line in enumerate(lines):
        line_no = i + 1

        # Skip comment lines
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # Skip if suppression present
        if has_ulp_suppression(content, i, lines):
            continue

        # Check for sub-ULP gap assertions: `> 1.0 - 1e-N` with N >= 8
        # Only flag if the line references a saturation function or assert
        if "assert" in line and SAT_FN_PATTERN.search(line):
            for m in ULP_TIGHT_GAP_PATTERN.finditer(line):
                exp_str = m.group(1)
                exp = int(exp_str) if exp_str[0] != "0" else int(exp_str)
                epsilon = 10**-exp
                errors.append(
                    f"  {c('FAIL', Colors.RED)}  {filepath.name}:{line_no}: "
                    f"ULP saturation violation: gap = 1e-{exp} ({epsilon:.1e}) < half-ULP(1.0) ≈ 6e-8\n"
                    f"{' ' * 10}In float32, sigmoid/tanh at saturated inputs rounds exactly to 1.0;\n"
                    f"{' ' * 10}demanding > 1-1e-{exp} is impossible because 1.0 - 1e-{exp} cannot be\n"
                    f"{' ' * 10}represented as a float32 distinct from 1.0.\n"
                    f"{' ' * 10}Fix: use `== 1.0` for exact saturation, or `> 0.9999999` for near-1.\n"
                    f"{' ' * 10}     Add '# ulp-ok: <justification>' to suppress if intentional."
                )

            for m in ULP_BORDERLINE_GAP_PATTERN.finditer(line):
                warnings.append(
                    f"  {c('WARN', Colors.YELLOW)}  {filepath.name}:{line_no}: "
                    f"Borderline ULP gap: `> 1 - 1e-7` ≈ ULP(1.0); "
                    f"may be flaky depending on rounding. Consider `> 0.9999999` or `== 1.0`."
                )

            # Check for `!= 1.0` / `!= 1` on lines calling sigmoid/tanh/exp
            # (handles nested parentheses like sigmoid(np.float32(80.0)))
            neq_one = re.search(r"!=\s*1(?:\.0+)?\b(?!\s*[-+*/])", line)
            if neq_one:
                for fn_name in ("sigmoid", "tanh", "exp"):
                    if re.search(rf"(?<![a-zA-Z0-9]){fn_name}\s*\(", line):
                        errors.append(
                            f"  {c('FAIL', Colors.RED)}  {filepath.name}:{line_no}: "
                            f"ULP saturation violation: `{fn_name}(...) != 1.0` in saturation region\n"
                            f"{' ' * 10}At large positive inputs, {fn_name} rounds exactly to 1.0 in float32;\n"
                            f"{' ' * 10}`!= 1.0` will be False for saturated inputs. Use `== 1.0` to verify\n"
                            f"{' ' * 10}saturation, or avoid testing exact inequality in saturation region."
                        )
                        break

            # Check for `!= 0.0` / `!= 0` on lines calling sigmoid
            neq_zero = re.search(r"!=\s*0(?:\.0+)?\b(?!\s*[-+*/])", line)
            if neq_zero and re.search(r"(?<![a-zA-Z0-9])sigmoid\s*\(", line):
                errors.append(
                    f"  {c('FAIL', Colors.RED)}  {filepath.name}:{line_no}: "
                    f"ULP saturation violation: `sigmoid(...) != 0.0` in deep negative region\n"
                    f"{' ' * 10}For x ≤ -88.73, exp(-x) overflows to inf, so sigmoid(x) = 1/(1+inf) = 0.0 exactly;\n"
                    f"{' ' * 10}`!= 0.0` will be False. Use `== 0.0` or test with non-saturating inputs."
                )

        # Check for impossibly small assertions (< 1e-46 below subnormal range)
        if "assert" in line:
            for m in ULP_IMPOSSIBLY_SMALL_PATTERN.finditer(line):
                errors.append(
                    f"  {c('FAIL', Colors.RED)}  {filepath.name}:{line_no}: "
                    f"ULP violation: assertion threshold below float32 subnormal minimum (≈1.4e-45)\n"
                    f"{' ' * 10}Float32 cannot represent values smaller than ~1.4e-45; such assertions\n"
                    f"{' ' * 10}will always pass vacuously or fail unpredictably."
                )

    return errors, warnings


# ── Per-file orchestration ────────────────────────────────────────────


def check_file(filepath: Path, verbose: bool = False) -> tuple[list[str], list[str]]:
    """Check a single test file for C¹ kink and ULP saturation violations.

    Returns (errors, warnings) lists of human-readable messages.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Only check Python test files
    if not filepath.name.startswith("test_") or not filepath.suffix == ".py":
        return errors, warnings

    # Skip exempt filenames (kink-specific tests)
    if is_exempt_filename(filepath):
        if verbose:
            warnings.append(f"  {c('SKIP', Colors.CYAN)}  {filepath.name} (exempt: kink stability test)")
        return errors, warnings

    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        warnings.append(f"  {c('WARN', Colors.YELLOW)}  {filepath.name}: cannot read ({e})")
        return errors, warnings

    # ── ULP saturation checks (run independently of kink checks) ──
    ulp_errors, ulp_warnings = check_ulp_saturation(content, filepath)
    errors.extend(ulp_errors)
    warnings.extend(ulp_warnings)

    # ── C¹ kink protection checks ──
    if has_c1_suppression(content):
        if verbose:
            warnings.append(f"  {c('SUPP', Colors.CYAN)}  {filepath.name} (C¹ kink suppressed by comment)")
    else:
        # Find C¹-discontinuous layers
        discontinuous_layers = find_c1_discontinuous_layers(content)
        if not discontinuous_layers:
            # Check for standard ReLU (informational only)
            if STANDARD_RELU_PATTERN.search(content) and has_numerical_gradient_check(content):
                has_prot = has_kink_protection(content)
                if not has_prot:
                    relu_lines = [m.start() for m in STANDARD_RELU_PATTERN.finditer(content)]
                    relu_line = content[: relu_lines[0]].count("\n") + 1 if relu_lines else 0
                    warnings.append(
                        f"  {c('WARN', Colors.YELLOW)}  {filepath.name}:{relu_line}: "
                        f"ReLU (negative_slope=0) numerical gradient uses offset strategy; "
                        f"consider using avoid_c1_discontinuity for rigor"
                    )
        else:
            # Check if there's a numerical gradient test in this file
            grad_lines = has_numerical_gradient_check(content)
            if grad_lines:
                # Check if protection is applied
                if has_kink_protection(content):
                    if verbose:
                        layer_desc = ", ".join(name for name, _ in discontinuous_layers)
                        errors.append(
                            f"  {c('PASS', Colors.GREEN)}  {filepath.name}: {layer_desc} numerical gradient protected"
                        )
                else:
                    # VIOLATION: C¹-discontinuous layer + numerical grad check, but no protection
                    for layer_name, line_no in discontinuous_layers:
                        grad_line_str = ", ".join(str(gl) for gl in grad_lines[:3])
                        errors.append(
                            f"  {c('FAIL', Colors.RED)}  {filepath.name}:{line_no}: "
                            f"{c(layer_name, Colors.BOLD)} numerical gradient check (line {grad_line_str}) "
                            f"missing C¹ kink protection!\n"
                            f"{' ' * 10}Fix: call avoid_c1_discontinuity(x, h=EPS) before the gradient check,\n"
                            f'{" " * 10}     or add "# c1-kink-ok: <justification>" to suppress.'
                        )

    return errors, warnings


def scan_directory(
    directory: Path,
    verbose: bool = False,
) -> tuple[int, int, list[str]]:
    """Scan all test files in a directory.

    Returns (error_count, warning_count, all_messages).
    """
    errors: list[str] = []
    warnings: list[str] = []
    files_checked = 0
    files_with_issues = 0

    test_files = sorted(directory.rglob("test_*.py"))

    for filepath in test_files:
        # Skip __pycache__ and hidden dirs
        if any(part.startswith((".", "__pycache__")) for part in filepath.parts):
            continue

        file_errors, file_warnings = check_file(filepath, verbose=verbose)
        files_checked += 1

        # Separate PASS messages from errors
        real_errors = [e for e in file_errors if "FAIL" in e]
        pass_messages = [e for e in file_errors if "PASS" in e]

        if real_errors:
            files_with_issues += 1
            errors.extend(real_errors)
        if verbose:
            errors.extend(pass_messages)
        warnings.extend(file_warnings)

    return files_checked, files_with_issues, errors + warnings


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check C¹ kink protection and ULP saturation safety in activation function tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_c1_kink_protection.py tests/python/
  python scripts/check_c1_kink_protection.py tests/python/ -v
  python scripts/check_c1_kink_protection.py tests/python/test_activation_backward.py
""",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="tests/python",
        help="Directory or file to check (default: tests/python/)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show pass messages and exempt files",
    )
    args = parser.parse_args()

    target = Path(args.path)

    if not target.exists():
        print(c(f"Error: path not found: {target}", Colors.RED), file=sys.stderr)
        return 2

    if target.is_file():
        errors, warnings = check_file(target, verbose=args.verbose)
        real_errors = [e for e in errors if "FAIL" in e]
        pass_messages = [e for e in errors if "PASS" in e]
        files_checked = 1
        files_with_issues = 1 if real_errors else 0
        messages = ([] if not args.verbose else pass_messages) + real_errors + warnings
    else:
        files_checked, files_with_issues, messages = scan_directory(target, verbose=args.verbose)

    # ── Report ─────────────────────────────────────────────────────
    print()
    print(c("═" * 70, Colors.BOLD))
    print(c("  Numerical Testing Safety Check (C¹ Kink + ULP Saturation)", Colors.BOLD + Colors.CYAN))
    print(c("═" * 70, Colors.BOLD))
    print()

    if messages:
        for msg in messages:
            print(msg)
        print()

    # Summary
    error_count = sum(1 for m in messages if "FAIL" in m)
    warning_count = sum(1 for m in messages if "WARN" in m or "SKIP" in m or "SUPP" in m)
    c1_errors = sum(1 for m in messages if "C¹ kink protection" in m or "C1" in m)
    ulp_errors = sum(1 for m in messages if "ULP" in m and "FAIL" in m)

    print(c("─" * 70, Colors.BOLD))
    print(f"  Files checked: {files_checked}")
    print(f"  Files with issues: {files_with_issues}")
    if error_count:
        print(
            f"  {c(f'Errors: {error_count}', Colors.RED)}"
            + (f" (C¹ kink: {c1_errors}, ULP: {ulp_errors})" if (c1_errors and ulp_errors) else "")
        )
    if warning_count:
        print(f"  {c(f'Warnings: {warning_count}', Colors.YELLOW)}")
    if error_count == 0:
        print(f"  {c('✓ All checks passed', Colors.GREEN)}")
    print(c("─" * 70, Colors.BOLD))
    print()

    if error_count > 0:
        print(c("  See .agents/docs/knowledge/best-practices/float-precision-testing-guide.md", Colors.CYAN))
        print(c("  §1 for ULP saturation rules, §2.6-2.8 for C¹ kink helper usage.", Colors.CYAN))
        print()

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
