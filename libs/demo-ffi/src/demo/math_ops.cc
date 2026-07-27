// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/string.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace demo_ffi {
namespace math_ops {

int64_t fibonacci(int64_t n) {
  if (n < 0) return -1;
  if (n <= 1) return n;
  int64_t a = 0, b = 1;
  for (int64_t i = 2; i <= n; ++i) {
    int64_t c = a + b;
    a = b;
    b = c;
  }
  return b;
}

bool is_prime(int64_t n) {
  if (n < 2) return false;
  if (n < 4) return true;
  if (n % 2 == 0 || n % 3 == 0) return false;
  for (int64_t i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) return false;
  }
  return true;
}

tvm::ffi::Array<double> vec_add(const tvm::ffi::Array<double>& a,
                                const tvm::ffi::Array<double>& b) {
  if (a.size() != b.size()) {
    TVM_FFI_THROW(tvm::ffi::ValueError)
        << "vec_add: input vectors must have same length, got "
        << a.size() << " vs " << b.size();
  }
  tvm::ffi::Array<double> result;
  result.reserve(a.size());
  for (int64_t i = 0; i < a.size(); ++i) {
    result.push_back(a[i] + b[i]);
  }
  return result;
}

tvm::ffi::Array<double> vec_scale(const tvm::ffi::Array<double>& v, double factor) {
  tvm::ffi::Array<double> result;
  result.reserve(v.size());
  for (int64_t i = 0; i < v.size(); ++i) {
    result.push_back(v[i] * factor);
  }
  return result;
}

double vec_dot(const tvm::ffi::Array<double>& a, const tvm::ffi::Array<double>& b) {
  if (a.size() != b.size()) {
    TVM_FFI_THROW(tvm::ffi::ValueError)
        << "vec_dot: input vectors must have same length, got "
        << a.size() << " vs " << b.size();
  }
  double sum = 0.0;
  for (int64_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

double vec_l2_norm(const tvm::ffi::Array<double>& v) {
  double sum_sq = 0.0;
  for (int64_t i = 0; i < v.size(); ++i) {
    sum_sq += v[i] * v[i];
  }
  return std::sqrt(sum_sq);
}

tvm::ffi::Tuple<double, double, double, double> vec_stats(
    const tvm::ffi::Array<double>& v) {
  if (v.empty()) {
    TVM_FFI_THROW(tvm::ffi::ValueError) << "vec_stats: input vector is empty";
  }
  double min_v = v[0], max_v = v[0];
  double sum = 0.0;
  for (int64_t i = 0; i < v.size(); ++i) {
    double x = v[i];
    if (x < min_v) min_v = x;
    if (x > max_v) max_v = x;
    sum += x;
  }
  double mean = sum / static_cast<double>(v.size());
  double var_sum = 0.0;
  for (int64_t i = 0; i < v.size(); ++i) {
    double diff = v[i] - mean;
    var_sum += diff * diff;
  }
  double stddev = std::sqrt(var_sum / static_cast<double>(v.size()));
  return tvm::ffi::Tuple<double, double, double, double>(min_v, max_v, mean, stddev);
}

int64_t count_substring(const tvm::ffi::String& text, const tvm::ffi::String& sub) {
  std::string s = text.c_str();
  std::string needle = sub.c_str();
  if (needle.empty()) return 0;
  int64_t count = 0;
  size_t pos = 0;
  while ((pos = s.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

tvm::ffi::String reverse_string(const tvm::ffi::String& s) {
  std::string str = s.c_str();
  std::reverse(str.begin(), str.end());
  return tvm::ffi::String(str);
}

int64_t gcd(int64_t a, int64_t b) {
  a = std::llabs(a);
  b = std::llabs(b);
  while (b != 0) {
    int64_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

int64_t lcm(int64_t a, int64_t b) {
  if (a == 0 || b == 0) return 0;
  return std::llabs(a / gcd(a, b) * b);
}

tvm::ffi::Array<int64_t> sieve_primes(int64_t limit) {
  if (limit < 2) return tvm::ffi::Array<int64_t>();
  std::vector<bool> is_p(static_cast<size_t>(limit + 1), true);
  is_p[0] = false;
  is_p[1] = false;
  for (int64_t i = 2; i * i <= limit; ++i) {
    if (is_p[static_cast<size_t>(i)]) {
      for (int64_t j = i * i; j <= limit; j += i) {
        is_p[static_cast<size_t>(j)] = false;
      }
    }
  }
  tvm::ffi::Array<int64_t> result;
  for (int64_t i = 2; i <= limit; ++i) {
    if (is_p[static_cast<size_t>(i)]) {
      result.push_back(i);
    }
  }
  return result;
}

double sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

tvm::ffi::Array<double> vec_sigmoid(const tvm::ffi::Array<double>& v) {
  tvm::ffi::Array<double> result;
  result.reserve(v.size());
  for (int64_t i = 0; i < v.size(); ++i) {
    result.push_back(sigmoid(v[i]));
  }
  return result;
}

int64_t binary_search(const tvm::ffi::Array<double>& sorted_arr, double target) {
  int64_t lo = 0, hi = sorted_arr.size() - 1;
  while (lo <= hi) {
    int64_t mid = lo + (hi - lo) / 2;
    double val = sorted_arr[mid];
    if (std::abs(val - target) < 1e-12) return mid;
    if (val < target) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("math.fibonacci", fibonacci)
      .def("math.is_prime", is_prime)
      .def("math.vec_add", vec_add)
      .def("math.vec_scale", vec_scale)
      .def("math.vec_dot", vec_dot)
      .def("math.vec_l2_norm", vec_l2_norm)
      .def("math.vec_stats", vec_stats)
      .def("math.count_substring", count_substring)
      .def("math.reverse_string", reverse_string)
      .def("math.gcd", gcd)
      .def("math.lcm", lcm)
      .def("math.sieve_primes", sieve_primes)
      .def("math.sigmoid", sigmoid)
      .def("math.vec_sigmoid", vec_sigmoid)
      .def("math.binary_search", binary_search);
}

}  // namespace math_ops
}  // namespace demo_ffi
