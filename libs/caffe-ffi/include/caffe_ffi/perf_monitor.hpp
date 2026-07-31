#ifndef CAFFE_FFI_PERF_MONITOR_HPP_
#define CAFFE_FFI_PERF_MONITOR_HPP_

/// @file perf_monitor.hpp
/// @brief 统一性能监控工具：RAII阶段计时器 + 值域/范数reduce辅助函数
///
/// 使用本头文件遵循 [单次遍历性能统计日志埋点] 模式：
///   1. 逐元素算子：计算+统计单次遍历融合
///   2. GEMM类算子：阶段级计时 + GEMM后独立reduce
///   3. 日志使用 [TAG-PERF] 结构化标签，字段k=v格式，time=Xus
///
/// @see docs/retrospective/patterns/code-patterns/single-pass-perf-instrumentation.md
/// @see .agents/checklists/framework-extension-and-perf-logging-review.md

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include "caffe_ffi/log.hpp"

namespace caffe_ffi {
namespace perf {

// ============================================================================
// 类型别名
// ============================================================================
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

// ============================================================================
// 值域统计器：在线累积min/max，支持跨batch running聚合
// ============================================================================
struct RangeStats {
  float min_val;
  float max_val;

  RangeStats()
      : min_val(std::numeric_limits<float>::max()),
        max_val(-std::numeric_limits<float>::max()) {}

  void Update(float v) {
    if (v < min_val) min_val = v;
    if (v > max_val) max_val = v;
  }

  void Merge(const RangeStats& other) {
    if (other.min_val < min_val) min_val = other.min_val;
    if (other.max_val > max_val) max_val = other.max_val;
  }

  void Reset() {
    min_val = std::numeric_limits<float>::max();
    max_val = -std::numeric_limits<float>::max();
  }

  bool Valid() const {
    return min_val != std::numeric_limits<float>::max();
  }

  std::string ToString() const {
    std::ostringstream oss;
    oss << "[" << min_val << ", " << max_val << "]";
    return oss.str();
  }
};

// ============================================================================
// 范数统计器：在线计算L2范数
// ============================================================================
struct NormStats {
  double sum_sq;

  NormStats() : sum_sq(0.0) {}

  void Update(float v) {
    sum_sq += static_cast<double>(v) * static_cast<double>(v);
  }

  float L2Norm() const {
    return static_cast<float>(std::sqrt(sum_sq));
  }

  void Reset() { sum_sq = 0.0; }
};

// ============================================================================
// RAII阶段计时器：构造时开始，析构时自动累积到指定double变量
// ============================================================================
class ScopedTimer {
 public:
  explicit ScopedTimer(double& accum_us)
      : accum_us_(accum_us), start_(Clock::now()) {}

  ~ScopedTimer() {
    auto end = Clock::now();
    accum_us_ += std::chrono::duration<double, std::micro>(end - start_).count();
  }

  // 禁止拷贝/赋值
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  double& accum_us_;
  TimePoint start_;
};

// ============================================================================
// 总计时器：构造时开始，ElapsedUs()返回耗时
// ============================================================================
class TotalTimer {
 public:
  TotalTimer() : start_(Clock::now()) {}

  double ElapsedUs() const {
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start_).count();
  }

  void Reset() { start_ = Clock::now(); }

 private:
  TimePoint start_;
};

// ============================================================================
// 独立Reduce辅助函数（GEMM后/计算后统计，纯读cache友好）
// ============================================================================

/// @brief 对float数组做值域reduce，返回RangeStats
inline RangeStats ReduceRange(const float* data, int64_t count) {
  RangeStats stats;
  for (int64_t i = 0; i < count; ++i) {
    stats.Update(data[i]);
  }
  return stats;
}

/// @brief 对float数组做值域+L2范数reduce
inline void ReduceRangeAndNorm(const float* data, int64_t count,
                                RangeStats& range, NormStats& norm) {
  range.Reset();
  norm.Reset();
  for (int64_t i = 0; i < count; ++i) {
    float v = data[i];
    range.Update(v);
    norm.Update(v);
  }
}

/// @brief Softmax概率分布统计：avg_max_prob + avg_entropy
/// @param data softmax输出数据，布局 [outer_num, channels, inner_num] (axis=1)
struct ProbStats {
  float avg_max_prob;
  float avg_entropy;
  RangeStats range;
};

inline ProbStats ReduceSoftmaxStats(const float* data, int outer_num,
                                     int channels, int inner_num) {
  ProbStats ps;
  ps.range.Reset();
  double sum_max = 0.0;
  double sum_entropy = 0.0;
  int n_samples = outer_num * inner_num;
  int dim = channels * inner_num;

  for (int i = 0; i < outer_num; ++i) {
    const float* slice = data + i * dim;
    for (int k = 0; k < inner_num; ++k) {
      float sample_max = 0.0f;
      double sample_entropy = 0.0;
      for (int j = 0; j < channels; ++j) {
        float p = slice[j * inner_num + k];
        ps.range.Update(p);
        if (p > sample_max) sample_max = p;
        if (p > 0.0f) {
          sample_entropy -= static_cast<double>(p) * std::log(static_cast<double>(p));
        }
      }
      sum_max += static_cast<double>(sample_max);
      sum_entropy += sample_entropy;
    }
  }
  ps.avg_max_prob = static_cast<float>(sum_max / static_cast<double>(n_samples));
  ps.avg_entropy = static_cast<float>(sum_entropy / static_cast<double>(n_samples));
  return ps;
}

// ============================================================================
// 日志输出辅助：拼接k=v字段
// ============================================================================
class PerfLogBuilder {
 public:
  explicit PerfLogBuilder(const char* tag, const char* layer_name,
                          const char* layer_type, const char* direction) {
    oss_ << "[" << tag << "] " << layer_name
         << " " << layer_type << " " << direction << ":";
  }

  PerfLogBuilder& Field(const char* key, int value) {
    oss_ << " " << key << "=" << value;
    return *this;
  }

  PerfLogBuilder& Field(const char* key, int64_t value) {
    oss_ << " " << key << "=" << value;
    return *this;
  }

  PerfLogBuilder& Field(const char* key, double value) {
    oss_ << " " << key << "=" << value;
    return *this;
  }

  PerfLogBuilder& Field(const char* key, float value) {
    oss_ << " " << key << "=" << value;
    return *this;
  }

  PerfLogBuilder& Field(const char* key, bool value) {
    oss_ << " " << key << "=" << (value ? "true" : "false");
    return *this;
  }

  PerfLogBuilder& Field(const char* key, const char* value) {
    oss_ << " " << key << "=" << value;
    return *this;
  }

  PerfLogBuilder& FieldRange(const char* key, const RangeStats& stats) {
    oss_ << " " << key << "=[" << stats.min_val << ", " << stats.max_val << "]";
    return *this;
  }

  PerfLogBuilder& FieldUs(const char* key, double value_us) {
    oss_ << " " << key << "=" << value_us << "us";
    return *this;
  }

  void EmitInfo() {
    CAFFE_FFI_LOG_INFO() << oss_.str();
  }

 private:
  std::ostringstream oss_;
};

// ============================================================================
// 标准化极值初始化宏（避免忘记正确初始化值）
// ============================================================================
#define CAFFE_FFI_PERF_RANGE_STATS(var_name) \
  ::caffe_ffi::perf::RangeStats var_name

#define CAFFE_FFI_PERF_NORM_STATS(var_name) \
  ::caffe_ffi::perf::NormStats var_name

#define CAFFE_FFI_PERF_SCOPED_TIMER(accum_us) \
  ::caffe_ffi::perf::ScopedTimer _scoped_timer_##__LINE__(accum_us)

#define CAFFE_FFI_PERF_TOTAL_TIMER(name) \
  ::caffe_ffi::perf::TotalTimer name

}  // namespace perf
}  // namespace caffe_ffi

#endif  // CAFFE_FFI_PERF_MONITOR_HPP_
