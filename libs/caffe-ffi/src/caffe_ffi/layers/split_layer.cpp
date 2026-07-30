#include "caffe_ffi/layers/split_layer.hpp"

#include <chrono>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void SplitLayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  int count = bottom[0]->count();
  int num_top = static_cast<int>(top.size());

  std::ostringstream shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) shape_ss << ", ";
    shape_ss << bottom[0]->shape(i);
  }

  CAFFE_FFI_LAYER_LOG << "Split Reshape: bottom shape=[" << shape_ss.str() << "]"
                      << " count=" << count
                      << " num_top=" << num_top
                      << " copy_bytes=" << (count * static_cast<int64_t>(sizeof(float)));

  // Measure total reshape (memory allocation) time
  auto t_reshape_start = std::chrono::high_resolution_clock::now();
  int64_t total_alloc_bytes = 0;

  for (int i = 0; i < num_top; ++i) {
    auto t_top_start = std::chrono::high_resolution_clock::now();
    int64_t bytes_before = top[i]->count() * static_cast<int64_t>(sizeof(float));
    top[i]->ReshapeLike(*bottom[0]);
    int64_t bytes_after = top[i]->count() * static_cast<int64_t>(sizeof(float));
    auto t_top_end = std::chrono::high_resolution_clock::now();
    double top_reshape_us = std::chrono::duration<double, std::micro>(
        t_top_end - t_top_start).count();
    total_alloc_bytes += (bytes_after - bytes_before);
    CAFFE_FFI_LAYER_LOG << "Split Reshape: top[" << i << "] reshape done"
                        << " bytes_before=" << bytes_before
                        << " bytes_after=" << bytes_after
                        << " reshape_time=" << top_reshape_us << "us";
  }

  auto t_reshape_end = std::chrono::high_resolution_clock::now();
  double reshape_ms = std::chrono::duration<double, std::milli>(
      t_reshape_end - t_reshape_start).count();

  CAFFE_FFI_LOG_WARN() << "[SPLIT-PERF] " << this->name()
                       << " Reshape: num_top=" << num_top
                       << " count=" << count
                       << " elem_size=" << sizeof(float) << "B"
                       << " total_copied_per_fwd=" << (num_top * count * static_cast<int64_t>(sizeof(float))) << "B"
                       << " reshape_time=" << reshape_ms << "ms"
                       << " net_alloc=" << total_alloc_bytes << "B";
}

void SplitLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  int count = bottom[0]->count();
  int num_top = static_cast<int>(top.size());
  int64_t copy_bytes_per_top = count * static_cast<int64_t>(sizeof(float));
  int64_t total_copy_bytes = num_top * copy_bytes_per_top;

  CAFFE_FFI_LAYER_LOG << "Split Forward: count=" << count
                      << " num_top=" << num_top
                      << " copy_bytes_per_top=" << copy_bytes_per_top
                      << " total_copy_bytes=" << total_copy_bytes
                      << " bottom_ptr=" << static_cast<const void*>(bottom_data);

  if (num_top == 1) {
    // N=1: identity passthrough (still copy for consistency/safety; zero-copy Future Work)
    auto t0 = std::chrono::high_resolution_clock::now();
    float* top_data = top[0]->cpu_data();
    if (top_data != bottom_data) {
      std::memcpy(top_data, bottom_data, copy_bytes_per_top);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double copy_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double throughput_gbs = (copy_bytes_per_top / (copy_us / 1e6)) / (1024.0 * 1024.0 * 1024.0);
    CAFFE_FFI_LOG_WARN() << "[SPLIT-PERF] " << this->name()
                         << " Forward(N=1): count=" << count
                         << " copied=" << copy_bytes_per_top << "B"
                         << " memcpy_time=" << copy_us << "us"
                         << " throughput=" << throughput_gbs << "GB/s"
                         << " inplace=" << (top_data == bottom_data ? "yes" : "no");
    CAFFE_FFI_LAYER_LOG << "Split Forward(N=1): top[0] ptr=" << static_cast<void*>(top_data)
                        << " memcpy completed";
    return;
  }

  // N >= 2: copy to each top
  auto t_total_start = std::chrono::high_resolution_clock::now();
  double max_copy_us = 0;
  double min_copy_us = 1e18;
  int alloc_count = 0;

  for (int i = 0; i < num_top; ++i) {
    float* top_data = top[i]->cpu_data();
    bool inplace = (top_data == bottom_data);

    auto t_copy_start = std::chrono::high_resolution_clock::now();
    if (!inplace) {
      std::memcpy(top_data, bottom_data, copy_bytes_per_top);
    } else {
      // Shouldn't normally happen for N>=2, but guard against it
      CAFFE_FFI_LOG_WARN() << "[SPLIT-PERF] " << this->name()
                           << " Forward: top[" << i << "] is in-place with bottom, skipping copy";
    }
    auto t_copy_end = std::chrono::high_resolution_clock::now();
    double copy_us = std::chrono::duration<double, std::micro>(
        t_copy_end - t_copy_start).count();

    max_copy_us = std::max(max_copy_us, copy_us);
    min_copy_us = std::min(min_copy_us, copy_us);
    if (!inplace) ++alloc_count;

    CAFFE_FFI_LAYER_LOG << "Split Forward: top[" << i << "] ptr=" << static_cast<void*>(top_data)
                        << " copied=" << copy_bytes_per_top << "B"
                        << " time=" << copy_us << "us"
                        << " inplace=" << (inplace ? "yes" : "no");
  }

  auto t_total_end = std::chrono::high_resolution_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(
      t_total_end - t_total_start).count();
  double avg_copy_us = total_ms * 1000.0 / std::max(1, num_top);
  double throughput_gbs = (total_copy_bytes / (total_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

  CAFFE_FFI_LOG_WARN() << "[SPLIT-PERF] " << this->name()
                       << " Forward(N=" << num_top << "): count=" << count
                       << " total_copied=" << total_copy_bytes << "B"
                       << " total_memcpy_time=" << total_ms << "ms"
                       << " avg_per_copy=" << avg_copy_us << "us"
                       << " min_copy=" << min_copy_us << "us"
                       << " max_copy=" << max_copy_us << "us"
                       << " throughput=" << throughput_gbs << "GB/s"
                       << " num_copies=" << alloc_count;
}

REGISTER_LAYER_CLASS(Split);

}  // namespace caffe_ffi
