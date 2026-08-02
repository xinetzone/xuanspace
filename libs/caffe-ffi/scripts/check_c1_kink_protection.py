#!/usr/bin/env python3
"""
C¹ kink protection checker for activation function numerical gradient tests.

Scans Python test files to ensure that numerical gradient tests for
C¹-discontinuous activation functions (LeakyReLU, PReLU) properly push
test points away from the kink at x=0 using `avoid_c1_discontinuity()`.

Rationale:
    Central differences across C¹-discontinuous kinks produce O(1) errors
    (derivative jump magnitude), not O(h²). This causes flaky tests when
    random inputs happen to land near the kink. The `avoid_c1_discontinuity`
    helper pushes points within 2h of the kink to a safe distance, ensuring
    [x-h, x+h] never straddles the discontinuity.

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

# ── Pattern definitions ────────────────────────────────────────────────

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
SUPPRESSION_PATTERNS: list[re.Pattern] = [
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


# ── Core checking logic ────────────────────────────────────────────────


def is_exempt_filename(filepath: Path) -> bool:
    """Check if a filename matches exempt patterns (kink validation tests)."""
    name = filepath.name
    return any(p.search(name) for p in EXEMPT_FILENAME_PATTERNS)


def has_suppression_comment(content: str) -> bool:
    """Check if the file has a file-level suppression comment."""
    # Check first 20 lines for file-level suppression
    lines = content.splitlines()[:20]
    header = "\n".join(lines)
    return any(p.search(header) for p in SUPPRESSION_PATTERNS)


def find_c1_discontinuous_layers(content: str) -> list[tuple[str, int]]:
    """Find all C¹-discontinuous layer references in the file.

    Returns list of (layer_type_name, line_number) tuples.
    """
    findings: list[tuple[str, int]] = []
    seen_lines: set[int] = set()

    for layer_name, pattern, _is_critical in C1_DISCONTINUOUS_PATTERNS:
        for match in pattern.finditer(content):
            # Find line number of this match
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


def check_file(filepath: Path, verbose: bool = False) -> tuple[list[str], list[str]]:
    """Check a single test file for C¹ kink protection violations.

    Returns (errors, warnings) lists of human-readable messages.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Only check Python test files
    if not filepath.name.startswith("test_") or not filepath.suffix == ".py":
        return errors, warnings

    # Skip exempt filenames
    if is_exempt_filename(filepath):
        if verbose:
            warnings.append(f"  {c('SKIP', Colors.CYAN)}  {filepath.name} (exempt: kink stability test)")
        return errors, warnings

    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        warnings.append(f"  {c('WARN', Colors.YELLOW)}  {filepath.name}: cannot read ({e})")
        return errors, warnings

    # Check for file-level suppression
    if has_suppression_comment(content):
        if verbose:
            warnings.append(f"  {c('SUPP', Colors.CYAN)}  {filepath.name} (suppressed by comment)")
        return errors, warnings

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
        return errors, warnings

    # Check if there's a numerical gradient test in this file
    grad_lines = has_numerical_gradient_check(content)
    if not grad_lines:
        return errors, warnings  # Layer exists but no numerical gradient testing

    # Check if protection is applied
    if has_kink_protection(content):
        if verbose:
            layer_desc = ", ".join(name for name, _ in discontinuous_layers)
            errors.append(f"  {c('PASS', Colors.GREEN)}  {filepath.name}: {layer_desc} numerical gradient protected")
        return errors, warnings

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
        description="Check C¹ kink protection in activation function numerical gradient tests.",
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
    print(c("  C¹ Kink Protection Check", Colors.BOLD + Colors.CYAN))
    print(c("═" * 70, Colors.BOLD))
    print()

    if messages:
        for msg in messages:
            print(msg)
        print()

    # Summary
    error_count = sum(1 for m in messages if "FAIL" in m)
    warning_count = sum(1 for m in messages if "WARN" in m or "SKIP" in m or "SUPP" in m)

    print(c("─" * 70, Colors.BOLD))
    print(f"  Files checked: {files_checked}")
    print(f"  Files with issues: {files_with_issues}")
    if error_count:
        print(f"  {c(f'Errors: {error_count}', Colors.RED)}")
    if warning_count:
        print(f"  {c(f'Warnings: {warning_count}', Colors.YELLOW)}")
    if error_count == 0:
        print(f"  {c('✓ All C¹ kink protection checks passed', Colors.GREEN)}")
    print(c("─" * 70, Colors.BOLD))
    print()

    if error_count > 0:
        print(c("  See .agents/docs/knowledge/best-practices/float-precision-testing-guide.md", Colors.CYAN))
        print(c("  §2.6-2.8 for helper function usage and strategy decision tree.", Colors.CYAN))
        print()

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
