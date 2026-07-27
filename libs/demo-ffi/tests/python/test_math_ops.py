"""Unit tests for the math_ops C++ accelerated module."""
from __future__ import annotations

import math

import pytest

from demo_ffi import demo
from demo_ffi.demo import math as m


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

    def test_fib_negative_returns_neg_one(self):
        assert m.fibonacci(-1) == -1


class TestIsPrime:
    @pytest.mark.parametrize("n,expected", [
        (0, False),
        (1, False),
        (2, True),
        (3, True),
        (4, False),
        (5, True),
        (9, False),
        (17, True),
        (97, True),
        (100, False),
        (101, True),
    ])
    def test_is_prime_values(self, n, expected):
        assert m.is_prime(n) is expected


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


class TestVecScale:
    def test_vec_scale_basic(self):
        assert m.vec_scale([1.0, 2.0, 3.0], 2.0) == pytest.approx([2.0, 4.0, 6.0])

    def test_vec_scale_zero(self):
        assert m.vec_scale([1.0, 2.0, 3.0], 0.0) == pytest.approx([0.0, 0.0, 0.0])

    def test_vec_scale_negative(self):
        assert m.vec_scale([1.0, -2.0], -1.0) == pytest.approx([-1.0, 2.0])


class TestVecDot:
    def test_dot_basic(self):
        assert m.vec_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)

    def test_dot_orthogonal(self):
        assert m.vec_dot([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_dot_mismatch_raises(self):
        with pytest.raises(ValueError):
            m.vec_dot([1.0], [1.0, 2.0])


class TestVecL2Norm:
    def test_norm_unit(self):
        assert m.vec_l2_norm([1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_norm_345(self):
        assert m.vec_l2_norm([3.0, 4.0]) == pytest.approx(5.0)

    def test_norm_zero(self):
        assert m.vec_l2_norm([0.0, 0.0]) == pytest.approx(0.0)


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


class TestStringOps:
    def test_count_substring_basic(self):
        assert m.count_substring("hello hello world", "hello") == 2

    def test_count_substring_overlap_not_counted(self):
        assert m.count_substring("aaa", "aa") == 1

    def test_count_substring_empty(self):
        assert m.count_substring("anything", "") == 0

    def test_reverse_string(self):
        assert m.reverse_string("hello") == "olleh"

    def test_reverse_empty(self):
        assert m.reverse_string("") == ""

    def test_reverse_palindrome(self):
        assert m.reverse_string("racecar") == "racecar"


class TestGcdLcm:
    @pytest.mark.parametrize("a,b,expected", [
        (12, 8, 4),
        (7, 5, 1),
        (0, 5, 5),
        (-12, 8, 4),
        (100, 75, 25),
    ])
    def test_gcd(self, a, b, expected):
        assert m.gcd(a, b) == expected

    @pytest.mark.parametrize("a,b,expected", [
        (4, 6, 12),
        (7, 5, 35),
        (0, 5, 0),
        (12, 18, 36),
    ])
    def test_lcm(self, a, b, expected):
        assert m.lcm(a, b) == expected


class TestSievePrimes:
    def test_sieve_small(self):
        assert m.sieve_primes(10) == [2, 3, 5, 7]

    def test_sieve_under_two(self):
        assert m.sieve_primes(1) == []
        assert m.sieve_primes(0) == []

    def test_sieve_includes_two(self):
        assert m.sieve_primes(2) == [2]

    def test_sieve_first_few(self):
        assert m.sieve_primes(20) == [2, 3, 5, 7, 11, 13, 17, 19]


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


class TestVecSigmoid:
    def test_vec_sigmoid(self):
        out = m.vec_sigmoid([0.0, 100.0, -100.0])
        assert out[0] == pytest.approx(0.5)
        assert out[1] == pytest.approx(1.0, abs=1e-10)
        assert out[2] == pytest.approx(0.0, abs=1e-10)

    def test_vec_sigmoid_length_preserved(self):
        v = [1.0, 2.0, 3.0, 4.0]
        assert len(m.vec_sigmoid(v)) == len(v)


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


class TestModuleExports:
    def test_demo_module_has_math(self):
        assert hasattr(demo, "math")
        assert demo.math is m

    def test_demo_core_functions_still_work(self):
        h = demo.tls_command_handle()
        assert h > 0
        demo.runtime_shutdown()
