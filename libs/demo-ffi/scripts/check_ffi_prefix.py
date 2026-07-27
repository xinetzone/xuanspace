#!/usr/bin/env python3
"""
FFI prefix consistency checker for tvm-ffi based projects.

Scans C++ source files and Python _ffi_api.py files to verify that:
1. All C++ registered prefixes have corresponding Python initialization
2. All Python initialized prefixes have corresponding C++ registrations
3. All registered functions are accessible via Python getattr (optional)
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @staticmethod
    def supports_color() -> bool:
        if os.environ.get("NO_COLOR"):
            return False
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def color(text: str, color_code: str) -> str:
    if Colors.supports_color():
        return f"{color_code}{text}{Colors.RESET}"
    return text


def find_project_root(start_path: Path | None = None) -> Path:
    """Find project root by looking for CMakeLists.txt and pyproject.toml."""
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current = start_path
    for _ in range(10):
        if (current / "CMakeLists.txt").exists() and (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    raise FileNotFoundError(
        "Could not find project root (must contain CMakeLists.txt and pyproject.toml)"
    )


def find_possible_lib_dirs(build_dir: Path) -> List[Path]:
    """Find all possible library output directories."""
    dirs = [build_dir / "lib"]
    src_dir = build_dir / "src"
    if src_dir.exists():
        for subdir in src_dir.iterdir():
            if subdir.is_dir():
                dirs.append(subdir)
                release_dir = subdir / "Release"
                if release_dir.exists():
                    dirs.append(release_dir)
    return dirs


def extract_cpp_functions(src_dir: Path, verbose: bool = False) -> Dict[str, List[str]]:
    """
    Extract all functions registered via tvm::ffi::reflection::GlobalDef() from C++ files.

    Returns a dict mapping prefix -> list of function names.
    """
    prefix_to_funcs: Dict[str, List[str]] = {}

    if not src_dir.exists():
        return prefix_to_funcs

    static_init_start = re.compile(r"TVM_FFI_STATIC_INIT_BLOCK\s*\(\s*\)\s*\{")
    def_pattern = re.compile(r'\.def\(\s*"([^"]+)"\s*,')

    for cc_file in sorted(src_dir.rglob("*.cc")):
        try:
            content = cc_file.read_text(encoding="utf-8")
        except Exception as e:
            if verbose:
                print(color(f"  Warning: Could not read {cc_file}: {e}", Colors.YELLOW))
            continue

        pos = 0
        while True:
            start_match = static_init_start.search(content, pos)
            if not start_match:
                break

            brace_start = content.find("{", start_match.start())
            if brace_start == -1:
                break

            depth = 0
            end_pos = brace_start
            for i in range(brace_start, len(content)):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end_pos = i
                        break

            block_content = content[brace_start + 1:end_pos]
            pos = end_pos + 1

            for def_match in def_pattern.finditer(block_content):
                func_name = def_match.group(1)
                if "." in func_name:
                    prefix = func_name.split(".")[0]
                else:
                    prefix = "_global"

                if prefix not in prefix_to_funcs:
                    prefix_to_funcs[prefix] = []
                prefix_to_funcs[prefix].append(func_name)

                if verbose:
                    print(f"  Found C++ function: {func_name} in {cc_file.name}")

    for prefix in prefix_to_funcs:
        prefix_to_funcs[prefix] = sorted(set(prefix_to_funcs[prefix]))

    return prefix_to_funcs


def extract_python_prefixes(python_dir: Path, verbose: bool = False) -> Dict[str, Tuple[Path, str]]:
    """
    Extract prefixes from _FFI_INIT_FUNC("prefix", __name__) calls in Python files.

    Returns a dict mapping prefix -> (file_path, python_module_path).
    The python_module_path is derived from the file's location relative to python/ dir.
    """
    prefix_to_info: Dict[str, Tuple[Path, str]] = {}

    if not python_dir.exists():
        return prefix_to_info

    init_pattern = re.compile(r'_FFI_INIT_FUNC\(\s*"([^"]+)"\s*,\s*__name__\s*\)')

    for ffi_file in sorted(python_dir.rglob("_ffi_api.py")):
        try:
            content = ffi_file.read_text(encoding="utf-8")
        except Exception as e:
            if verbose:
                print(color(f"  Warning: Could not read {ffi_file}: {e}", Colors.YELLOW))
            continue

        rel_path = ffi_file.relative_to(python_dir)
        parts = list(rel_path.parts)
        parts[-1] = parts[-1].replace(".py", "")
        module_path = ".".join(parts)

        for match in init_pattern.finditer(content):
            prefix = match.group(1)
            prefix_to_info[prefix] = (ffi_file, module_path)
            if verbose:
                print(f"  Found Python prefix: {prefix} in {ffi_file.name} -> module {module_path}")

    return prefix_to_info


def verify_python_imports(
    project_root: Path,
    prefixes: Set[str],
    cpp_funcs: Dict[str, List[str]],
    prefix_to_info: Dict[str, Tuple[Path, str]],
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Dynamically import _ffi_api modules and verify all functions are accessible.

    Groups prefixes by their containing _ffi_api.py module, imports each module once,
    and checks that all registered FFI functions exist as attributes on _LIB.

    Returns (all_passed, list_of_errors).
    """
    errors: List[str] = []
    build_dir = project_root / "build"
    lib_dirs = find_possible_lib_dirs(build_dir)

    python_path_added = False
    original_path = sys.path.copy()
    python_root = project_root / "python"
    if python_root.exists() and str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
        python_path_added = True

    for lib_dir in lib_dirs:
        if lib_dir.exists():
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(lib_dir))
            except (OSError, AttributeError):
                pass
            lib_str = str(lib_dir)
            if sys.platform == "win32":
                os.environ["PATH"] = lib_str + os.pathsep + os.environ.get("PATH", "")
            elif sys.platform == "darwin":
                os.environ["DYLD_LIBRARY_PATH"] = lib_str + os.pathsep + os.environ.get("DYLD_LIBRARY_PATH", "")
            else:
                os.environ["LD_LIBRARY_PATH"] = lib_str + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

    try:
        import importlib

        ffi_module_to_prefixes: Dict[str, List[str]] = {}
        for prefix in prefixes:
            if prefix in prefix_to_info:
                _, mod_path = prefix_to_info[prefix]
                if mod_path not in ffi_module_to_prefixes:
                    ffi_module_to_prefixes[mod_path] = []
                ffi_module_to_prefixes[mod_path].append(prefix)

        for ffi_mod_path, mod_prefixes in sorted(ffi_module_to_prefixes.items()):
            try:
                module = importlib.import_module(ffi_mod_path)
                if verbose:
                    print(f"  Imported {ffi_mod_path}")

                for prefix in mod_prefixes:
                    if prefix in cpp_funcs:
                        for func_name in cpp_funcs[prefix]:
                            attr_name = func_name.split(".", 1)[1] if "." in func_name else func_name
                            if not hasattr(module, attr_name):
                                errors.append(
                                    f"Function '{func_name}' not registered on "
                                    f"module {ffi_mod_path}"
                                )
                            elif verbose:
                                print(f"    Verified: {attr_name}")

                wrapper_pkg = ffi_mod_path.rsplit("._ffi_api", 1)[0]
                for prefix in mod_prefixes:
                    wrapper_mod_name = f"{wrapper_pkg}.{prefix}"
                    try:
                        wrapper_mod = importlib.import_module(wrapper_mod_name)
                        if verbose:
                            print(f"  Imported wrapper: {wrapper_mod_name}")
                        for func_name in cpp_funcs.get(prefix, []):
                            attr_name = func_name.split(".", 1)[1] if "." in func_name else func_name
                            if not hasattr(wrapper_mod, attr_name):
                                if verbose:
                                    print(f"    [INFO] Wrapper {wrapper_mod_name}.{attr_name} "
                                          f"not directly exposed (may be wrapped with Python logic)")
                    except ImportError:
                        if verbose:
                            print(f"  [INFO] No wrapper module: {wrapper_mod_name}")
            except ImportError as e:
                errors.append(f"Could not import {ffi_mod_path}: {e}")
            except Exception as e:
                errors.append(f"Error checking {ffi_mod_path}: {e}")

    finally:
        if python_path_added:
            sys.path = original_path

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(
        description="Check FFI prefix consistency between C++ and Python"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Path to project root (default: auto-detect)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output detailed matching information",
    )
    parser.add_argument(
        "--skip-dynamic",
        action="store_true",
        help="Skip dynamic import verification (useful when build not available)",
    )
    args = parser.parse_args()

    print("")
    print(color("========================================", Colors.CYAN))
    print(color("  FFI Prefix Consistency Checker", Colors.CYAN))
    print(color("========================================", Colors.CYAN))
    print("")

    try:
        project_root = args.project_root if args.project_root else find_project_root()
    except FileNotFoundError as e:
        print(color(f"[FAIL] {e}", Colors.RED))
        return 1

    print(f"Project root: {project_root}")
    print("")

    src_dir = project_root / "src"
    python_dir = project_root / "python"

    step_color = Colors.CYAN
    pass_color = Colors.GREEN
    fail_color = Colors.RED
    warn_color = Colors.YELLOW

    # Step 1: Extract C++ registrations
    print(color("==> Scanning C++ source files...", step_color))
    cpp_prefix_funcs = extract_cpp_functions(src_dir, args.verbose)
    cpp_prefixes = set(cpp_prefix_funcs.keys())
    total_cpp_funcs = sum(len(funcs) for funcs in cpp_prefix_funcs.values())
    print(color(f"    Found {total_cpp_funcs} registered functions in {len(cpp_prefixes)} prefix(es)", pass_color))
    if args.verbose:
        for prefix in sorted(cpp_prefixes):
            print(f"    - {prefix}: {len(cpp_prefix_funcs[prefix])} functions")
    print("")

    # Step 2: Extract Python initializations
    print(color("==> Scanning Python _ffi_api.py files...", step_color))
    python_prefix_info = extract_python_prefixes(python_dir, args.verbose)
    python_prefixes = set(python_prefix_info.keys())
    print(color(f"    Found {len(python_prefixes)} initialized prefix(es)", pass_color))
    if args.verbose:
        for prefix in sorted(python_prefixes):
            fpath, modpath = python_prefix_info[prefix]
            print(f"    - {prefix} ({fpath.name} -> {modpath})")
    print("")

    # Step 3: Check prefix consistency
    all_passed = True
    warnings = []

    print(color("==> Checking prefix consistency...", step_color))

    missing_in_python = cpp_prefixes - python_prefixes
    missing_in_cpp = python_prefixes - cpp_prefixes
    matching_prefixes = cpp_prefixes & python_prefixes

    if missing_in_python:
        all_passed = False
        print(color(f"    [FAIL] MISSING IN PYTHON: {len(missing_in_python)} prefix(es)", fail_color))
        for prefix in sorted(missing_in_python):
            print(f"      - {prefix} ({len(cpp_prefix_funcs[prefix])} C++ functions)")
    else:
        print(color(f"    [PASS] All C++ prefixes initialized in Python", pass_color))

    if missing_in_cpp:
        msg = f"    MISSING IN C++: {len(missing_in_cpp)} prefix(es)"
        if args.strict:
            all_passed = False
            print(color(f"    [FAIL] {msg}", fail_color))
        else:
            warnings.append(f"Python prefixes without C++ registrations: {missing_in_cpp}")
            print(color(f"    [WARN] {msg}", warn_color))
        for prefix in sorted(missing_in_cpp):
            fpath, _ = python_prefix_info[prefix]
            print(f"      - {prefix} ({fpath.name})")
    else:
        print(color(f"    [PASS] All Python prefixes have C++ registrations", pass_color))

    if matching_prefixes:
        print(color(f"    [INFO] Matching prefixes: {len(matching_prefixes)}", Colors.CYAN))
        for prefix in sorted(matching_prefixes):
            print(f"      - {prefix} ({len(cpp_prefix_funcs[prefix])} functions)")
    print("")

    # Step 4: Dynamic import verification
    if args.skip_dynamic:
        print(color("==> Skipping dynamic import verification (--skip-dynamic)", step_color))
        print("")
    else:
        print(color("==> Verifying dynamic imports...", step_color))
        try:
            imports_ok, import_errors = verify_python_imports(
                project_root, matching_prefixes, cpp_prefix_funcs,
                python_prefix_info, args.verbose
            )
            if imports_ok:
                print(color(f"    [PASS] All functions registered and accessible", pass_color))
            else:
                all_passed = False
                print(color(f"    [FAIL] {len(import_errors)} import/access error(s)", fail_color))
                for err in import_errors:
                    print(f"      - {err}")
        except Exception as e:
            msg = f"    Could not perform dynamic verification: {e}"
            if args.strict:
                all_passed = False
                print(color(f"    [FAIL] {msg}", fail_color))
            else:
                warnings.append(str(e))
                print(color(f"    [WARN] {msg}", warn_color))
        print("")

    # Step 5: Summary
    print(color("========================================", Colors.BOLD))
    print(color("  Summary", Colors.BOLD))
    print(color("========================================", Colors.BOLD))
    print(f"  Total C++ functions: {total_cpp_funcs}")
    print(f"  C++ prefixes:        {len(cpp_prefixes)}")
    print(f"  Python prefixes:     {len(python_prefixes)}")
    print(f"  Matching prefixes:   {len(matching_prefixes)}")
    if warnings:
        print(f"  Warnings:            {len(warnings)}")
    print("")

    if all_passed:
        print(color("  [PASS] All checks passed!", pass_color))
        if warnings and not args.strict:
            print(color(f"  ({len(warnings)} warning(s) - use --strict to fail on warnings)", warn_color))
        print("")
        return 0
    else:
        print(color("  [FAIL] Some checks failed", fail_color))
        print("")
        return 1


if __name__ == "__main__":
    sys.exit(main())
