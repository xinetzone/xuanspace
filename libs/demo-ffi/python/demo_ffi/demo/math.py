"""Math operations module with C++ accelerated implementations via FFI."""
from __future__ import annotations

from typing import List, Tuple

from . import _ffi_api as _LIB


def _value_error_call(fn, *args):
    """Call an FFI function and convert RuntimeError to ValueError for user errors."""
    try:
        return fn(*args)
    except RuntimeError as e:
        msg = str(e)
        raise ValueError(msg) from None


def fibonacci(n: int) -> int:
    """Compute the nth Fibonacci number (iterative, O(n)).

    Args:
        n: Non-negative integer index (0-based).

    Returns:
        The nth Fibonacci number.
    """
    return int(_LIB.fibonacci(int(n)))


def is_prime(n: int) -> bool:
    """Check if n is a prime number using trial division with 6k±1 optimization.

    Args:
        n: Integer to test.

    Returns:
        True if n is prime.
    """
    return bool(_LIB.is_prime(int(n)))


def vec_add(a: List[float], b: List[float]) -> List[float]:
    """Element-wise addition of two vectors.

    Args:
        a: First vector (list of floats).
        b: Second vector (list of floats).

    Returns:
        New vector with a[i] + b[i] for each position.

    Raises:
        ValueError: If vectors have different lengths.
    """
    return list(_value_error_call(_LIB.vec_add, list(a), list(b)))


def vec_scale(v: List[float], factor: float) -> List[float]:
    """Multiply each element of vector v by factor.

    Args:
        v: Input vector.
        factor: Scalar multiplier.

    Returns:
        New vector with v[i] * factor.
    """
    return list(_LIB.vec_scale(list(v), float(factor)))


def vec_dot(a: List[float], b: List[float]) -> float:
    """Compute the dot product of two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Scalar dot product sum(a[i] * b[i]).

    Raises:
        ValueError: If vectors have different lengths.
    """
    return float(_value_error_call(_LIB.vec_dot, list(a), list(b)))


def vec_l2_norm(v: List[float]) -> float:
    """Compute the L2 (Euclidean) norm of a vector.

    Args:
        v: Input vector.

    Returns:
        sqrt(sum(x_i^2)).
    """
    return float(_LIB.vec_l2_norm(list(v)))


def vec_stats(v: List[float]) -> Tuple[float, float, float, float]:
    """Compute basic descriptive statistics of a vector.

    Args:
        v: Non-empty vector of numbers.

    Returns:
        Tuple of (min, max, mean, stddev).

    Raises:
        ValueError: If v is empty.
    """
    result = _value_error_call(_LIB.vec_stats, list(v))
    return (float(result[0]), float(result[1]), float(result[2]), float(result[3]))


def count_substring(text: str, sub: str) -> int:
    """Count non-overlapping occurrences of sub in text.

    Args:
        text: Haystack string.
        sub: Needle substring.

    Returns:
        Number of occurrences.
    """
    return int(_LIB.count_substring(str(text), str(sub)))


def reverse_string(s: str) -> str:
    """Reverse a string.

    Args:
        s: Input string.

    Returns:
        Reversed string.
    """
    return str(_LIB.reverse_string(str(s)))


def gcd(a: int, b: int) -> int:
    """Compute the greatest common divisor of a and b (non-negative).

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        GCD absolute value.
    """
    return int(_LIB.gcd(int(a), int(b)))


def lcm(a: int, b: int) -> int:
    """Compute the least common multiple of a and b.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        LCM (non-negative), or 0 if either input is 0.
    """
    return int(_LIB.lcm(int(a), int(b)))


def sieve_primes(limit: int) -> List[int]:
    """Generate all prime numbers up to `limit` using the Sieve of Eratosthenes.

    Args:
        limit: Upper bound (inclusive). Must be >= 2.

    Returns:
        List of primes in increasing order.
    """
    return list(_LIB.sieve_primes(int(limit)))


def sigmoid(x: float) -> float:
    """Compute the logistic sigmoid function 1/(1+exp(-x)).

    Args:
        x: Scalar input.

    Returns:
        Sigmoid value in (0, 1).
    """
    return float(_LIB.sigmoid(float(x)))


def vec_sigmoid(v: List[float]) -> List[float]:
    """Apply sigmoid element-wise to a vector.

    Args:
        v: Input vector.

    Returns:
        Vector of sigmoid values.
    """
    return list(_LIB.vec_sigmoid(list(v)))


def binary_search(sorted_arr: List[float], target: float) -> int:
    """Binary search for target in a sorted ascending array.

    Args:
        sorted_arr: Sorted list of floats (ascending order).
        target: Value to find.

    Returns:
        Index of target if found (within 1e-12 tolerance), otherwise -1.
    """
    return int(_LIB.binary_search(list(sorted_arr), float(target)))


__all__ = [
    "fibonacci",
    "is_prime",
    "vec_add",
    "vec_scale",
    "vec_dot",
    "vec_l2_norm",
    "vec_stats",
    "count_substring",
    "reverse_string",
    "gcd",
    "lcm",
    "sieve_primes",
    "sigmoid",
    "vec_sigmoid",
    "binary_search",
]
