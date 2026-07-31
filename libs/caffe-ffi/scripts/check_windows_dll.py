#!/usr/bin/env python3
"""
Windows DLL self-check script for caffe-ffi build verification.

Checks that all required DLLs are present in the build output directory
and that the caffe_ffi shared library can be loaded successfully.

Usage:
    python scripts/check_windows_dll.py                # auto-detect build dir
    python scripts/check_windows_dll.py --build-dir <path>
    python scripts/check_windows_dll.py --verbose
    python scripts/check_windows_dll.py --skip-load     # skip DLL loading test
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class Colors:
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


def find_project_root(start_path: Optional[Path] = None, verbose: bool = False) -> Path:
    """Find project root by looking for CMakeLists.txt and pyproject.toml."""
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    if verbose:
        print(f"  [CHECK-DLL] find_project_root: starting from {start_path}")

    current = start_path
    for i in range(10):
        has_cmake = (current / "CMakeLists.txt").exists()
        has_pyproject = (current / "pyproject.toml").exists()
        if verbose:
            print(f"  [CHECK-DLL]   [{i}] checking {current}: "
                  f"CMakeLists={'OK' if has_cmake else 'MISS'}, "
                  f"pyproject={'OK' if has_pyproject else 'MISS'}")
        if has_cmake and has_pyproject:
            if verbose:
                print(f"  [CHECK-DLL] find_project_root: found at {current} (level {i})")
            return current
        parent = current.parent
        if parent == current:
            if verbose:
                print(f"  [CHECK-DLL] find_project_root: reached filesystem root, stopping")
            break
        current = parent

    if verbose:
        print(f"  [CHECK-DLL] find_project_root: FAILED after {i+1} levels")
    raise FileNotFoundError(
        "Could not find project root (must contain CMakeLists.txt and pyproject.toml)"
    )


def find_dll_dirs(build_dir: Path, verbose: bool = False) -> List[Path]:
    """Find all directories that may contain DLLs in the build tree."""
    dirs: List[Path] = []
    lib_dir = build_dir / "lib"
    if lib_dir.exists():
        if verbose:
            print(f"  [CHECK-DLL] find_dll_dirs: + lib/ (exists)")
        dirs.append(lib_dir)
    elif verbose:
        print(f"  [CHECK-DLL] find_dll_dirs: - lib/ (not found)")

    src_dir = build_dir / "src"
    if src_dir.exists():
        if verbose:
            print(f"  [CHECK-DLL] find_dll_dirs: scanning src/ subdirectories...")
        for subdir in sorted(src_dir.iterdir()):
            if subdir.is_dir():
                dirs.append(subdir)
                if verbose:
                    print(f"  [CHECK-DLL] find_dll_dirs: + {subdir}")
                release_dir = subdir / "Release"
                if release_dir.exists():
                    dirs.append(release_dir)
                    if verbose:
                        print(f"  [CHECK-DLL] find_dll_dirs:   + {release_dir}")
    elif verbose:
        print(f"  [CHECK-DLL] find_dll_dirs: src/ not found, skipping")

    if verbose:
        print(f"  [CHECK-DLL] find_dll_dirs: total {len(dirs)} directory(s)")
    return dirs


# Required DLL patterns for caffe-ffi runtime
REQUIRED_DLL_PATTERNS = {
    "_caffe_ffi": ["_caffe_ffi.*"],  # main library
    "tvm_ffi": ["tvm_ffi.*"],
    "protobuf": ["libprotobuf*.dll", "libprotoc*.dll"],
    "abseil": ["absl_*.dll"],
    "utf8_range": ["utf8_range*.dll", "utf8_validity*.dll"],
    "openblas": ["libopenblas*.dll", "openblas*.dll"],
}

# Optional: MSVC runtime DLLs (may be system-installed)
OPTIONAL_DLL_PATTERNS = {
    "msvc_runtime": ["vcruntime*.dll", "msvcp*.dll", "concrt*.dll"],
    "msvc_ucrt": ["ucrtbase*.dll", "api-ms-win-*.dll"],
    "zlib": ["zlib*.dll"],  # indirect dependency via Protobuf; not required for core functionality
}


def scan_dlls(dll_dirs: List[Path], verbose: bool = False) -> Dict[str, List[Path]]:
    """Scan all DLL directories and return a dict of category -> list of found DLL paths."""
    found: Dict[str, List[Path]] = {cat: [] for cat in REQUIRED_DLL_PATTERNS}

    total_dlls = 0
    for dll_dir in dll_dirs:
        if not dll_dir.exists():
            if verbose:
                print(f"  [CHECK-DLL] scan_dlls: skip non-existent dir {dll_dir}")
            continue
        dll_files = list(dll_dir.rglob("*.dll"))
        if verbose:
            print(f"  [CHECK-DLL] scan_dlls: scanning {dll_dir} ({len(dll_files)} DLLs)")
        for dll_file in sorted(dll_files):
            total_dlls += 1
            fname = dll_file.name.lower()
            matched = False
            for category, patterns in REQUIRED_DLL_PATTERNS.items():
                for pattern in patterns:
                    p = pattern.lower().replace("*", "")
                    if p in fname or fname.startswith(p.rstrip(".")):
                        if dll_file not in found[category]:
                            found[category].append(dll_file)
                        if verbose:
                            print(f"  [CHECK-DLL] scan_dlls: [{category}] {dll_file.name}")
                        matched = True
                        break
                if matched:
                    break
            if verbose and not matched:
                print(f"  [CHECK-DLL] scan_dlls: [unmatched] {dll_file.name}")

    if verbose:
        total_matched = sum(len(v) for v in found.values())
        print(f"  [CHECK-DLL] scan_dlls: total {total_dlls} DLL(s) scanned, "
              f"{total_matched} matched to {len(REQUIRED_DLL_PATTERNS)} categories")

    return found


def check_dlls(found: Dict[str, List[Path]], verbose: bool = False) -> Tuple[bool, List[str]]:
    """Check that all required DLL categories have at least one match."""
    all_ok = True
    errors: List[str] = []

    if verbose:
        print(f"  [CHECK-DLL] check_dlls: checking {len(REQUIRED_DLL_PATTERNS)} required categories")

    for category, patterns in REQUIRED_DLL_PATTERNS.items():
        if found[category]:
            status = color(f"[PASS] {len(found[category])} file(s)", Colors.GREEN)
            if verbose:
                for f in found[category]:
                    print(f"    {f.name}")
        else:
            status = color("[FAIL] missing", Colors.RED)
            all_ok = False
            err_msg = f"Missing required DLL category: {category} (patterns: {patterns})"
            errors.append(err_msg)
            if verbose:
                print(f"  [CHECK-DLL] check_dlls: FAIL [{category}] patterns={patterns}, "
                      f"scanned categories with data: {[k for k, v in found.items() if v]}")
        print(f"  {status}  {category}")

    if verbose:
        print(f"  [CHECK-DLL] check_dlls: result={'ALL_PASS' if all_ok else 'HAS_FAILURES'}, "
              f"errors={len(errors)}")

    return all_ok, errors


def setup_dll_path(dll_dirs: List[Path], verbose: bool = False) -> None:
    """Add DLL directories to PATH and os.add_dll_directory()."""
    if verbose:
        print(f"  [CHECK-DLL] setup_dll_path: adding {len(dll_dirs)} dir(s) to DLL search path")
    for dll_dir in dll_dirs:
        if dll_dir.exists():
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(dll_dir))
                    if verbose:
                        print(f"  [CHECK-DLL] setup_dll_path: add_dll_directory({dll_dir})")
            except (OSError, AttributeError) as e:
                if verbose:
                    print(f"  [CHECK-DLL] setup_dll_path: add_dll_directory failed for {dll_dir}: {e}")
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
            if verbose:
                print(f"  [CHECK-DLL] setup_dll_path: prepended to PATH: {dll_dir}")
        elif verbose:
            print(f"  [CHECK-DLL] setup_dll_path: skip non-existent dir {dll_dir}")


def try_load_caffe_ffi(project_root: Path, dll_dirs: List[Path], verbose: bool = False) -> Tuple[bool, str]:
    """Try to import caffe_ffi to verify the DLL can be loaded."""
    setup_dll_path(dll_dirs, verbose=verbose)

    python_root = str(project_root / "python")
    original_path = sys.path.copy()
    path_added = False
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
        path_added = True
        if verbose:
            print(f"  [CHECK-DLL] try_load: added to sys.path: {python_root}")

    if verbose:
        kdup = os.environ.get("KMP_DUPLICATE_LIB_OK", "<not set>")
        print(f"  [CHECK-DLL] try_load: KMP_DUPLICATE_LIB_OK={kdup}")

    try:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        if verbose:
            print(f"  [CHECK-DLL] try_load: attempting import caffe_ffi...")
        import caffe_ffi  # type: ignore
        version = getattr(caffe_ffi, "__version__", "unknown")
        if verbose:
            import caffe_ffi as _cf
            print(f"  [CHECK-DLL] try_load: caffe_ffi imported from {getattr(_cf, '__file__', 'unknown')}")
        return True, f"caffe_ffi {version} loaded successfully"
    except ImportError as e:
        if verbose:
            import traceback
            print(f"  [CHECK-DLL] try_load: ImportError - {e}")
            traceback.print_exc()
        return False, f"Import failed: {e}"
    except Exception as e:
        if verbose:
            import traceback
            print(f"  [CHECK-DLL] try_load: Exception ({type(e).__name__}) - {e}")
            traceback.print_exc()
        return False, f"Load error: {e}"
    finally:
        if path_added:
            sys.path = original_path
            if verbose:
                print(f"  [CHECK-DLL] try_load: restored sys.path")


def check_dll_dependencies(dll_path: Path, verbose: bool = False) -> Tuple[bool, str]:
    """Check DLL dependencies using dumpbin (if available)."""
    import subprocess
    if verbose:
        print(f"  [CHECK-DLL] dumpbin: checking dependencies for {dll_path.name}")
    try:
        result = subprocess.run(
            ["dumpbin", "/dependents", str(dll_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            if verbose:
                print(f"  [CHECK-DLL] dumpbin: succeeded for {dll_path.name}")
            return True, result.stdout
        if verbose:
            print(f"  [CHECK-DLL] dumpbin: failed (rc={result.returncode}): {result.stderr[:200]}")
        return False, result.stderr or "dumpbin failed"
    except FileNotFoundError:
        if verbose:
            print(f"  [CHECK-DLL] dumpbin: dumpbin.exe not found in PATH")
        return False, "dumpbin not found (install VS Build Tools or Windows SDK)"
    except Exception as e:
        if verbose:
            print(f"  [CHECK-DLL] dumpbin: exception: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Windows DLL self-check for caffe-ffi build"
    )
    parser.add_argument(
        "--build-dir", type=Path, default=None,
        help="Path to build directory (default: auto-detect)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed scanning and loading information"
    )
    parser.add_argument(
        "--skip-load", action="store_true",
        help="Skip the caffe_ffi import loading test"
    )
    parser.add_argument(
        "--dumpbin", action="store_true",
        help="Run dumpbin /dependents on _caffe_ffi DLL (Windows SDK required)"
    )
    args = parser.parse_args()

    print("")
    print(color("=" * 48, Colors.CYAN))
    print(color("  caffe-ffi Windows DLL Self-Check", Colors.CYAN))
    print(color("=" * 48, Colors.CYAN))
    print("")

    if sys.platform != "win32":
        if args.verbose:
            print(f"  [CHECK-DLL] main: platform={sys.platform}, skipping (Windows-only)")
        print(color("[SKIP] This script is Windows-only.", Colors.YELLOW))
        print("")
        return 0

    if args.verbose:
        print(f"  [CHECK-DLL] main: platform={sys.platform}, python={sys.version.split()[0]}")

    try:
        project_root = find_project_root(verbose=args.verbose)
    except FileNotFoundError as e:
        if args.verbose:
            print(f"  [CHECK-DLL] main: find_project_root failed")
        print(color(f"[FAIL] {e}", Colors.RED))
        return 1

    print(f"Project root: {project_root}")
    build_dir = args.build_dir or (project_root / "build")
    print(f"Build dir:    {build_dir}")
    print("")

    if not build_dir.exists():
        if args.verbose:
            print(f"  [CHECK-DLL] main: build_dir={build_dir} does not exist, aborting")
        print(color("[FAIL] Build directory not found. Run cmake --build first.", Colors.RED))
        return 1

    # Step 1: Find DLL directories
    print(color("==> Scanning DLL directories...", Colors.CYAN))
    dll_dirs = find_dll_dirs(build_dir, verbose=args.verbose)
    if args.verbose:
        for d in dll_dirs:
            print(f"  {d}")
    print(f"  Found {len(dll_dirs)} DLL directory(s)")
    print("")

    # Step 2: Scan for required DLLs
    print(color("==> Checking required DLLs...", Colors.CYAN))
    found = scan_dlls(dll_dirs, verbose=args.verbose)
    dlls_ok, dll_errors = check_dlls(found, verbose=args.verbose)
    print("")

    # Step 3: Optional dumpbin check
    dumpbin_ok = True
    if args.dumpbin:
        print(color("==> Checking DLL dependencies (dumpbin)...", Colors.CYAN))
        caffe_dlls = found.get("_caffe_ffi", [])
        if caffe_dlls:
            dll_path = caffe_dlls[0]
            ok, output = check_dll_dependencies(dll_path, verbose=args.verbose)
            if ok:
                print(color(f"  [PASS] dumpbin succeeded for {dll_path.name}", Colors.GREEN))
                if args.verbose:
                    print(output[:2000])
            else:
                dumpbin_ok = False
                print(color(f"  [WARN] {output}", Colors.YELLOW))
        else:
            print(color("  [SKIP] _caffe_ffi DLL not found", Colors.YELLOW))
        print("")

    # Step 4: Try loading caffe_ffi
    load_ok = True
    if not args.skip_load:
        print(color("==> Testing caffe_ffi import...", Colors.CYAN))
        load_ok, load_msg = try_load_caffe_ffi(project_root, dll_dirs, verbose=args.verbose)
        if load_ok:
            print(color(f"  [PASS] {load_msg}", Colors.GREEN))
        else:
            print(color(f"  [FAIL] {load_msg}", Colors.RED))
        print("")

    # Summary
    print(color("=" * 48, Colors.BOLD))
    print(color("  Summary", Colors.BOLD))
    print(color("=" * 48, Colors.BOLD))
    passed = dlls_ok and load_ok and dumpbin_ok
    if args.verbose:
        print(f"  [CHECK-DLL] main: summary - dlls_ok={dlls_ok}, load_ok={load_ok}, "
              f"dumpbin_ok={dumpbin_ok}, final={passed}")
    if passed:
        print(color("  [PASS] All checks passed!", Colors.GREEN))
    else:
        if not dlls_ok:
            print(color(f"  [FAIL] DLL checks: {len(dll_errors)} issue(s)", Colors.RED))
            for err in dll_errors:
                print(f"    - {err}")
        if not load_ok:
            print(color("  [FAIL] Import test failed", Colors.RED))
        if not dumpbin_ok:
            print(color("  [WARN] dumpbin check had issues", Colors.YELLOW))
    print("")

    if args.verbose:
        print(f"  [CHECK-DLL] main: exit_code={0 if passed else 1}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())