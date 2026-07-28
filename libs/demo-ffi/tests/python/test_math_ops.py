"""Unit tests for the math_ops C++ accelerated module — including edge cases."""
from __future__ import annotations

import math
import random

import pytest

from demo_ffi import demo
from demo_ffi.demo import math as m

# ---------------------------------------------------------------------------
# 1. fibonacci
# ---------------------------------------------------------------------------

class TestFibonacci:
    def test_fib_base_cases(self):
        assert m.fibonacci(0) == 0
        assert m.fibonacci(1) == 1

    def test_fib_small(self):
        assert m.fibonacci(2) == 1
        assert m.fibonacci(5) == 5
        assert m.fibonacci(10) == 55

    def test_fib_larger(self):
        assert m.fibonacci(20) == 6765
        assert m.fibonacci(30) == 832040

    def test_fib_negative_returns_neg_one(self):
        assert m.fibonacci(-1) == -1
        assert m.fibonacci(-100) == -1

    @pytest.mark.parametrize("n,expected", [
        (0, 0), (1, 1), (2, 1), (3, 2),
        (10, 55), (20, 6765), (50, 12586269025),
    ])
    def test_fib_known_values(self, n, expected):
        assert m.fibonacci(n) == expected

    def test_fib_monotonic_for_n_ge_1(self):
        prev = m.fibonacci(1)
        for i in range(2, 40):
            cur = m.fibonacci(i)
            assert cur >= prev
            prev = cur


# ---------------------------------------------------------------------------
# 2. is_prime
# ---------------------------------------------------------------------------

class TestIsPrime:
    @pytest.mark.parametrize("n,expected", [
        (-5, False), (-1, False), (0, False), (1, False),
        (2, True), (3, True), (4, False), (5, True),
        (9, False), (17, True), (97, True), (100, False),
        (101, True),
    ])
    def test_is_prime_values(self, n, expected):
        assert m.is_prime(n) is expected

    @pytest.mark.parametrize("n", [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 997])
    def test_is_prime_large_primes(self, n):
        assert m.is_prime(n) is True

    @pytest.mark.parametrize("n", [4, 6, 8, 9, 10, 100, 1000, 999, 1001])
    def test_is_prime_composites(self, n):
        assert m.is_prime(n) is False

    def test_is_prime_even_numbers_gt2(self):
        for n in range(4, 100, 2):
            assert m.is_prime(n) is False

    def test_is_prime_int64_boundary_safe(self):
        assert m.is_prime(2**31 - 1) is True
        assert m.is_prime(2**31 - 2) is False


# ---------------------------------------------------------------------------
# 3. vec_add
# ---------------------------------------------------------------------------

class TestVecAdd:
    def test_vec_add_basic(self):
        assert m.vec_add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx([5.0, 7.0, 9.0])

    def test_vec_add_empty(self):
        assert m.vec_add([], []) == []

    def test_vec_add_single(self):
        assert m.vec_add([3.5], [1.5]) == pytest.approx([5.0])

    def test_vec_add_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            m.vec_add([1.0, 2.0], [1.0])

    def test_vec_add_negative_values(self):
        assert m.vec_add([-1.0, -2.0], [3.0, -4.0]) == pytest.approx([2.0, -6.0])

    def test_vec_add_zeros(self):
        assert m.vec_add([0.0, 0.0], [0.0, 0.0]) == pytest.approx([0.0, 0.0])

    def test_vec_add_large_vector_correctness(self):
        n = 10000
        a = [float(i) for i in range(n)]
        b = [float(i * 2) for i in range(n)]
        expected = [float(i * 3) for i in range(n)]
        result = m.vec_add(a, b)
        assert len(result) == n
        assert result[:10] == pytest.approx(expected[:10])
        assert result[-10:] == pytest.approx(expected[-10:])

    def test_vec_add_floating_point_precision(self):
        assert m.vec_add([0.1], [0.2])[0] == pytest.approx(0.3, abs=1e-10)


# ---------------------------------------------------------------------------
# 4. vec_scale
# ---------------------------------------------------------------------------

class TestVecScale:
    def test_vec_scale_basic(self):
        assert m.vec_scale([1.0, 2.0, 3.0], 2.0) == pytest.approx([2.0, 4.0, 6.0])

    def test_vec_scale_zero(self):
        assert m.vec_scale([1.0, 2.0, 3.0], 0.0) == pytest.approx([0.0, 0.0, 0.0])

    def test_vec_scale_negative(self):
        assert m.vec_scale([1.0, -2.0], -1.0) == pytest.approx([-1.0, 2.0])

    def test_vec_scale_by_one_is_identity(self):
        v = [3.14, -2.71, 0.0]
        assert m.vec_scale(v, 1.0) == pytest.approx(v)

    def test_vec_scale_empty(self):
        assert m.vec_scale([], 5.0) == []

    def test_vec_scale_large_factor(self):
        result = m.vec_scale([1.0], 1e10)
        assert result[0] == pytest.approx(1e10)

    def test_vec_scale_fractional(self):
        assert m.vec_scale([10.0, 20.0], 0.5) == pytest.approx([5.0, 10.0])


# ---------------------------------------------------------------------------
# 5. vec_dot
# ---------------------------------------------------------------------------

class TestVecDot:
    def test_dot_basic(self):
        assert m.vec_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)

    def test_dot_orthogonal(self):
        assert m.vec_dot([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_dot_mismatch_raises(self):
        with pytest.raises(ValueError):
            m.vec_dot([1.0], [1.0, 2.0])

    def test_dot_single_element(self):
        assert m.vec_dot([3.0], [7.0]) == pytest.approx(21.0)

    def test_dot_negative_values(self):
        assert m.vec_dot([-1.0, 2.0], [3.0, -4.0]) == pytest.approx(-11.0)

    def test_dot_empty_is_zero(self):
        assert m.vec_dot([], []) == pytest.approx(0.0)

    def test_dot_zero_vector(self):
        assert m.vec_dot([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_dot_commutative(self):
        a = [1.5, 2.5, 3.5]
        b = [4.0, 5.0, 6.0]
        assert m.vec_dot(a, b) == pytest.approx(m.vec_dot(b, a))


# ---------------------------------------------------------------------------
# 6. vec_l2_norm
# ---------------------------------------------------------------------------

class TestVecL2Norm:
    def test_norm_unit(self):
        assert m.vec_l2_norm([1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_norm_345(self):
        assert m.vec_l2_norm([3.0, 4.0]) == pytest.approx(5.0)

    def test_norm_zero(self):
        assert m.vec_l2_norm([0.0, 0.0]) == pytest.approx(0.0)

    def test_norm_empty_is_zero(self):
        assert m.vec_l2_norm([]) == pytest.approx(0.0)

    def test_norm_negative_values(self):
        assert m.vec_l2_norm([-3.0, -4.0]) == pytest.approx(5.0)

    def test_norm_single_element(self):
        assert m.vec_l2_norm([7.0]) == pytest.approx(7.0)
        assert m.vec_l2_norm([-7.0]) == pytest.approx(7.0)

    def test_norm_non_negativity(self):
        for _ in range(20):
            v = [random.uniform(-100, 100) for _ in range(50)]
            assert m.vec_l2_norm(v) >= 0.0


# ---------------------------------------------------------------------------
# 7. vec_stats
# ---------------------------------------------------------------------------

class TestVecStats:
    def test_stats_simple(self):
        mn, mx, mean, std = m.vec_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mn == pytest.approx(1.0)
        assert mx == pytest.approx(5.0)
        assert mean == pytest.approx(3.0)
        assert std == pytest.approx(math.sqrt(2.0))

    def test_stats_constant(self):
        mn, mx, mean, std = m.vec_stats([7.0, 7.0, 7.0])
        assert mn == pytest.approx(7.0)
        assert mx == pytest.approx(7.0)
        assert mean == pytest.approx(7.0)
        assert std == pytest.approx(0.0)

    def test_stats_empty_raises(self):
        with pytest.raises(ValueError):
            m.vec_stats([])

    def test_stats_single_element(self):
        mn, mx, mean, std = m.vec_stats([42.0])
        assert mn == pytest.approx(42.0)
        assert mx == pytest.approx(42.0)
        assert mean == pytest.approx(42.0)
        assert std == pytest.approx(0.0)

    def test_stats_negative_values(self):
        mn, mx, mean, std = m.vec_stats([-5.0, -1.0, -3.0])
        assert mn == pytest.approx(-5.0)
        assert mx == pytest.approx(-1.0)
        assert mean == pytest.approx(-3.0)

    def test_stats_mixed_signs(self):
        mn, mx, mean, std = m.vec_stats([-2.0, 2.0])
        assert mn == pytest.approx(-2.0)
        assert mx == pytest.approx(2.0)
        assert mean == pytest.approx(0.0)
        assert std == pytest.approx(2.0)

    def test_stats_large_vector(self):
        n = 100000
        v = [float(i) for i in range(n)]
        mn, mx, mean, std = m.vec_stats(v)
        assert mn == pytest.approx(0.0)
        assert mx == pytest.approx(float(n - 1))
        assert mean == pytest.approx((n - 1) / 2.0)

    def test_stats_return_type_is_tuple_of_four_floats(self):
        result = m.vec_stats([1.0, 2.0])
        assert len(result) == 4
        for val in result:
            assert isinstance(val, float)


# ---------------------------------------------------------------------------
# 8. count_substring
# ---------------------------------------------------------------------------

class TestCountSubstring:
    def test_count_substring_basic(self):
        assert m.count_substring("hello hello world", "hello") == 2

    def test_count_substring_overlap_not_counted(self):
        assert m.count_substring("aaa", "aa") == 1

    def test_count_substring_empty(self):
        assert m.count_substring("anything", "") == 0

    def test_count_substring_needle_longer_than_haystack(self):
        assert m.count_substring("hi", "hello world") == 0

    def test_count_substring_at_start(self):
        assert m.count_substring("foobar", "foo") == 1

    def test_count_substring_at_end(self):
        assert m.count_substring("foobar", "bar") == 1

    def test_count_substring_exact_match(self):
        assert m.count_substring("exact", "exact") == 1

    def test_count_substring_no_match(self):
        assert m.count_substring("abcdef", "xyz") == 0

    def test_count_substring_empty_haystack(self):
        assert m.count_substring("", "a") == 0
        assert m.count_substring("", "") == 0

    def test_count_substring_special_chars(self):
        assert m.count_substring("a.b.c.d", ".") == 3
        assert m.count_substring("  spaces  ", "  ") == 2

    def test_count_substring_case_sensitive(self):
        assert m.count_substring("Hello hello", "hello") == 1
        assert m.count_substring("Hello hello", "Hello") == 1


# ---------------------------------------------------------------------------
# 9. reverse_string
# ---------------------------------------------------------------------------

class TestReverseString:
    def test_reverse_string(self):
        assert m.reverse_string("hello") == "olleh"

    def test_reverse_empty(self):
        assert m.reverse_string("") == ""

    def test_reverse_palindrome(self):
        assert m.reverse_string("racecar") == "racecar"

    def test_reverse_single_char(self):
        assert m.reverse_string("x") == "x"

    def test_reverse_twice_is_identity(self):
        s = "The quick brown fox"
        assert m.reverse_string(m.reverse_string(s)) == s

    def test_reverse_with_spaces(self):
        assert m.reverse_string("a b c") == "c b a"

    def test_reverse_with_numbers(self):
        assert m.reverse_string("12345") == "54321"

    def test_reverse_special_chars(self):
        assert m.reverse_string("a!b@c#") == "#c@b!a"


# ---------------------------------------------------------------------------
# 10. gcd
# ---------------------------------------------------------------------------

class TestGcd:
    @pytest.mark.parametrize("a,b,expected", [
        (12, 8, 4),
        (7, 5, 1),
        (0, 5, 5),
        (-12, 8, 4),
        (100, 75, 25),
    ])
    def test_gcd(self, a, b, expected):
        assert m.gcd(a, b) == expected

    def test_gcd_zero_zero(self):
        assert m.gcd(0, 0) == 0

    def test_gcd_both_negative(self):
        assert m.gcd(-12, -8) == 4

    def test_gcd_identity(self):
        assert m.gcd(1, 100) == 1
        assert m.gcd(100, 1) == 1

    def test_gcd_same_number(self):
        assert m.gcd(42, 42) == 42

    def test_gcd_commutative(self):
        assert m.gcd(48, 18) == m.gcd(18, 48)

    def test_gcd_large_values(self):
        assert m.gcd(2**20, 2**15) == 2**15


# ---------------------------------------------------------------------------
# 11. lcm
# ---------------------------------------------------------------------------

class TestLcm:
    @pytest.mark.parametrize("a,b,expected", [
        (4, 6, 12),
        (7, 5, 35),
        (0, 5, 0),
        (12, 18, 36),
    ])
    def test_lcm(self, a, b, expected):
        assert m.lcm(a, b) == expected

    def test_lcm_with_one(self):
        assert m.lcm(1, 7) == 7
        assert m.lcm(7, 1) == 7

    def test_lcm_negative_numbers(self):
        assert m.lcm(-4, 6) == 12
        assert m.lcm(4, -6) == 12
        assert m.lcm(-4, -6) == 12

    def test_lcm_coprime(self):
        assert m.lcm(7, 11) == 77

    def test_lcm_same_number(self):
        assert m.lcm(5, 5) == 5

    def test_lcm_both_zero(self):
        assert m.lcm(0, 0) == 0

    def test_lcm_gcd_relation(self):
        for a, b in [(12, 8), (7, 5), (100, 75), (1, 99)]:
            assert m.lcm(a, b) * m.gcd(a, b) == abs(a * b)


# ---------------------------------------------------------------------------
# 12. sieve_primes
# ---------------------------------------------------------------------------

class TestSievePrimes:
    def test_sieve_small(self):
        assert m.sieve_primes(10) == [2, 3, 5, 7]

    def test_sieve_under_two(self):
        assert m.sieve_primes(1) == []
        assert m.sieve_primes(0) == []
        assert m.sieve_primes(-5) == []

    def test_sieve_includes_two(self):
        assert m.sieve_primes(2) == [2]

    def test_sieve_first_few(self):
        assert m.sieve_primes(20) == [2, 3, 5, 7, 11, 13, 17, 19]

    def test_sieve_count_matches_prime_pi(self):
        primes = m.sieve_primes(100)
        assert len(primes) == 25

    def test_sieve_all_returned_values_are_prime(self):
        primes = m.sieve_primes(500)
        for p in primes:
            assert m.is_prime(p), f"{p} should be prime"

    def test_sieve_returns_sorted_list(self):
        primes = m.sieve_primes(200)
        assert primes == sorted(primes)

    def test_sieve_no_duplicates(self):
        primes = m.sieve_primes(200)
        assert len(primes) == len(set(primes))

    def test_sieve_boundary_3(self):
        assert m.sieve_primes(3) == [2, 3]


# ---------------------------------------------------------------------------
# 13. sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_sigmoid_zero(self):
        assert m.sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        assert m.sigmoid(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_sigmoid_large_negative(self):
        assert m.sigmoid(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_sigmoid_symmetric(self):
        for x in [0.5, 1.0, 2.5]:
            assert m.sigmoid(x) + m.sigmoid(-x) == pytest.approx(1.0)

    @pytest.mark.parametrize("x,expected", [
        (1.0, 1.0 / (1.0 + math.exp(-1.0))),
        (-1.0, 1.0 / (1.0 + math.exp(1.0))),
        (2.0, 1.0 / (1.0 + math.exp(-2.0))),
    ])
    def test_sigmoid_known_values(self, x, expected):
        assert m.sigmoid(x) == pytest.approx(expected)

    def test_sigmoid_range(self):
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            result = m.sigmoid(x)
            assert 0.0 <= result <= 1.0

    def test_sigmoid_monotonically_increasing(self):
        xs = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
        results = [m.sigmoid(x) for x in xs]
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]

    def test_sigmoid_numerical_stability_large_x(self):
        assert m.sigmoid(500.0) == pytest.approx(1.0, abs=1e-10)
        assert m.sigmoid(-500.0) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 14. vec_sigmoid
# ---------------------------------------------------------------------------

class TestVecSigmoid:
    def test_vec_sigmoid(self):
        out = m.vec_sigmoid([0.0, 100.0, -100.0])
        assert out[0] == pytest.approx(0.5)
        assert out[1] == pytest.approx(1.0, abs=1e-10)
        assert out[2] == pytest.approx(0.0, abs=1e-10)

    def test_vec_sigmoid_length_preserved(self):
        v = [1.0, 2.0, 3.0, 4.0]
        assert len(m.vec_sigmoid(v)) == len(v)

    def test_vec_sigmoid_empty(self):
        assert m.vec_sigmoid([]) == []

    def test_vec_sigmoid_all_zeros(self):
        out = m.vec_sigmoid([0.0, 0.0, 0.0])
        assert out == pytest.approx([0.5, 0.5, 0.5])

    def test_vec_sigmoid_elementwise_matches_scalar(self):
        v = [-3.0, -1.0, 0.0, 1.0, 3.0]
        out = m.vec_sigmoid(v)
        for x, y in zip(v, out):
            assert y == pytest.approx(m.sigmoid(x))

    def test_vec_sigmoid_range(self):
        v = [random.uniform(-10, 10) for _ in range(100)]
        out = m.vec_sigmoid(v)
        for val in out:
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# 15. binary_search
# ---------------------------------------------------------------------------

class TestBinarySearch:
    def test_binary_search_found(self):
        arr = [1.0, 3.0, 5.0, 7.0, 9.0]
        assert m.binary_search(arr, 5.0) == 2

    def test_binary_search_first(self):
        arr = [1.0, 3.0, 5.0]
        assert m.binary_search(arr, 1.0) == 0

    def test_binary_search_last(self):
        arr = [1.0, 3.0, 5.0]
        assert m.binary_search(arr, 5.0) == 2

    def test_binary_search_not_found(self):
        arr = [1.0, 3.0, 5.0]
        assert m.binary_search(arr, 4.0) == -1

    def test_binary_search_empty_array(self):
        assert m.binary_search([], 1.0) == -1

    def test_binary_search_single_element_found(self):
        assert m.binary_search([42.0], 42.0) == 0

    def test_binary_search_single_element_not_found(self):
        assert m.binary_search([42.0], 1.0) == -1

    def test_binary_search_duplicates_returns_one_match(self):
        arr = [1.0, 2.0, 2.0, 2.0, 3.0]
        idx = m.binary_search(arr, 2.0)
        assert 0 <= idx < len(arr)
        assert arr[idx] == pytest.approx(2.0)

    def test_binary_search_negative_values(self):
        arr = [-10.0, -5.0, 0.0, 5.0, 10.0]
        assert m.binary_search(arr, -5.0) == 1
        assert m.binary_search(arr, 0.0) == 2
        assert m.binary_search(arr, 7.0) == -1

    def test_binary_search_large_sorted_array(self):
        n = 100000
        arr = [float(i) for i in range(n)]
        assert m.binary_search(arr, 50000.0) == 50000
        assert m.binary_search(arr, -1.0) == -1
        assert m.binary_search(arr, float(n)) == -1

    def test_binary_search_float_precision(self):
        arr = [0.1, 0.2, 0.3, 0.4, 0.5]
        idx = m.binary_search(arr, 0.3)
        assert idx == 2

    def test_binary_search_matches_all_elements(self):
        arr = [float(i * 0.5) for i in range(20)]
        for i, val in enumerate(arr):
            assert m.binary_search(arr, val) == i


# ---------------------------------------------------------------------------
# Cross-function / integration tests
# ---------------------------------------------------------------------------

class TestCrossFunctionIntegration:
    def test_dot_norm_relation(self):
        """|v|^2 == dot(v, v)"""
        v = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert m.vec_l2_norm(v) ** 2 == pytest.approx(m.vec_dot(v, v))

    def test_scale_then_dot(self):
        """dot(scale(v, a), scale(v, b)) == a*b*dot(v,v)"""
        v = [1.0, 2.0, 3.0]
        a, b = 2.0, 3.0
        sv_a = m.vec_scale(v, a)
        sv_b = m.vec_scale(v, b)
        assert m.vec_dot(sv_a, sv_b) == pytest.approx(a * b * m.vec_dot(v, v))

    def test_primes_up_to_all_pass_is_prime(self):
        """Every prime found by sieve must be confirmed prime by is_prime."""
        primes = m.sieve_primes(5000)
        for p in primes:
            assert m.is_prime(p) is True

    def test_add_inverse(self):
        """vec_add(a, scale(a, -1)) == zeros"""
        a = [1.0, -2.0, 3.5]
        result = m.vec_add(a, m.vec_scale(a, -1.0))
        assert result == pytest.approx([0.0, 0.0, 0.0])


class TestPerfLogDoesNotCrash:
    """Verify that enabling perf logging does not crash any function."""

    def test_perf_log_vec_stats(self, monkeypatch):
        monkeypatch.setenv("DEMO_FFI_PERF_LOG", "1")
        mn, mx, mean, std = m.vec_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mn == pytest.approx(1.0)
        assert mx == pytest.approx(5.0)
        assert mean == pytest.approx(3.0)

    def test_perf_log_sigmoid(self, monkeypatch):
        monkeypatch.setenv("DEMO_FFI_PERF_LOG", "1")
        assert m.sigmoid(0.0) == pytest.approx(0.5)
        out = m.vec_sigmoid([0.0, 1.0, -1.0])
        assert len(out) == 3


class TestModuleExports:
    def test_demo_module_has_math(self):
        assert hasattr(demo, "math")
        assert demo.math is m

    def test_demo_core_functions_still_work(self):
        h = demo.tls_command_handle()
        assert h > 0
        demo.runtime_shutdown()
