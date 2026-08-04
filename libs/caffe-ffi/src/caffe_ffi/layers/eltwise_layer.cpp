#include "caffe_ffi/layers/eltwise_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"

namespace caffe_ffi {

void EltwiseLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                               const std::vector<Blob*>& top) {
  const caffe::EltwiseParameter& param = this->layer_param_.eltwise_param();
  op_ = static_cast<EltwiseOp>(param.operation());
  CAFFE_FFI_CHECK_VALUE(op_ == PROD || op_ == SUM || op_ == MAX)
      << "EltwiseLayer only supports PROD, SUM, and MAX operations.";

  const char* op_name = "UNKNOWN";
  switch (op_) {
    case PROD: op_name = "PROD"; break;
    case SUM: op_name = "SUM"; break;
    case MAX: op_name = "MAX"; break;
  }

  const int num_bottoms = static_cast<int>(bottom.size());
  coeffs_.resize(num_bottoms, 1.0f);
  if (param.coeff_size() > 0) {
    CAFFE_FFI_CHECK_VALUE_EQ(param.coeff_size(), num_bottoms)
        << "EltwiseLayer coeff count must match bottom count.";
    for (int i = 0; i < num_bottoms; ++i) {
      coeffs_[i] = param.coeff(i);
    }
  }

  std::ostringstream coeffs_ss;
  for (int i = 0; i < num_bottoms; ++i) {
    if (i > 0) coeffs_ss << ", ";
    coeffs_ss << coeffs_[i];
  }

  CAFFE_FFI_LAYER_LOG << "Eltwise LayerSetUp: op=" << op_name
                      << " num_bottoms=" << num_bottoms
                      << " coeffs=[" << coeffs_ss.str() << "]";
}

void EltwiseLayer::Reshape(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  // Helper lambda to format a blob's shape as "(d0, d1, ..., dn)"
  auto shape_str = [](const Blob* b) -> std::string {
    std::ostringstream oss;
    oss << "(";
    for (int i = 0; i < b->num_axes(); ++i) {
      if (i > 0) oss << ", ";
      oss << b->shape(i);
    }
    oss << ")";
    return oss.str();
  };

  for (size_t i = 1; i < bottom.size(); ++i) {
    // Check number of axes first
    if (bottom[i]->num_axes() != bottom[0]->num_axes()) {
      std::ostringstream available_shapes;
      for (size_t k = 0; k < bottom.size(); ++k) {
        if (k > 0) available_shapes << "; ";
        available_shapes << "bottom[" << k << "] shape=" << shape_str(bottom[k])
                         << " ndim=" << bottom[k]->num_axes();
      }
      CAFFE_FFI_LOG_ERROR() << "[ELTWISE-SHAPE-MISMATCH] layer='" << this->name()
                            << "' op=" << (op_ == SUM ? "SUM" : op_ == PROD ? "PROD" : "MAX")
                            << " axis_count_mismatch: bottom[0] has " << bottom[0]->num_axes()
                            << " axes but bottom[" << i << "] has " << bottom[i]->num_axes()
                            << " axes."
                            << " All bottom shapes: " << available_shapes.str()
                            << " HINT: caffe-ffi Eltwise requires EXACT shape match (no broadcasting)."
                            << " If you intended broadcasting (e.g. PE (1,S,D) + input (N,S,D)),"
                            << " pre-broadcast the smaller tensor to match the larger shape in numpy"
                            << " before feeding to the network, or use Bias/Scale layers for"
                            << " per-channel/per-dimension operations.";
      CAFFE_FFI_CHECK_VALUE_EQ(bottom[i]->num_axes(), bottom[0]->num_axes())
          << "All bottom blobs must have the same number of axes (layer '"
          << this->name() << "'). See [ELTWISE-SHAPE-MISMATCH] above.";
    }
    // Check each axis dimension
    for (int j = 0; j < bottom[0]->num_axes(); ++j) {
      if (bottom[i]->shape(j) != bottom[0]->shape(j)) {
        std::ostringstream dim_detail;
        for (int k = 0; k < bottom[0]->num_axes(); ++k) {
          if (k > 0) dim_detail << ", ";
          dim_detail << "axis " << k << ": bottom[0]=" << bottom[0]->shape(k)
                     << " bottom[" << i << "]=" << bottom[i]->shape(k);
          if (bottom[i]->shape(k) != bottom[0]->shape(k)) {
            dim_detail << " [MISMATCH]";
          }
        }
        CAFFE_FFI_LOG_ERROR() << "[ELTWISE-SHAPE-MISMATCH] layer='" << this->name()
                              << "' op=" << (op_ == SUM ? "SUM" : op_ == PROD ? "PROD" : "MAX")
                              << " dimension_mismatch at axis=" << j
                              << ": bottom[0].shape(" << j << ")=" << bottom[0]->shape(j)
                              << " but bottom[" << i << "].shape(" << j << ")=" << bottom[i]->shape(j)
                              << ". bottom[0]=" << shape_str(bottom[0])
                              << " bottom[" << i << "]=" << shape_str(bottom[i])
                              << ". All axis details: " << dim_detail.str()
                              << " HINT: caffe-ffi Eltwise requires EXACT shape match (no broadcasting)."
                              << " If one dimension is 1 and you want numpy-style broadcasting,"
                              << " pre-broadcast the smaller tensor in numpy before feeding to the network.";
        CAFFE_FFI_CHECK_VALUE_EQ(bottom[i]->shape(j), bottom[0]->shape(j))
            << "All bottom blobs must have the same shape (layer '" << this->name()
            << "', axis " << j << "). See [ELTWISE-SHAPE-MISMATCH] above.";
      }
    }
  }
  top[0]->ReshapeLike(*bottom[0]);

  // Allocate max_idx_ buffer for MAX operation (winner-take-all gradient routing)
  if (op_ == MAX) {
    max_idx_.resize(static_cast<size_t>(bottom[0]->count()));
  } else {
    max_idx_.clear();
  }

  std::ostringstream input_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) input_shape_ss << ", ";
    input_shape_ss << bottom[0]->shape(i);
  }
  std::ostringstream output_shape_ss;
  for (int i = 0; i < top[0]->num_axes(); ++i) {
    if (i > 0) output_shape_ss << ", ";
    output_shape_ss << top[0]->shape(i);
  }

  const char* op_name = "UNKNOWN";
  switch (op_) {
    case PROD: op_name = "PROD"; break;
    case SUM: op_name = "SUM"; break;
    case MAX: op_name = "MAX"; break;
  }

  CAFFE_FFI_LAYER_LOG << "Eltwise Reshape: op=" << op_name
                      << " num_bottoms=" << bottom.size()
                      << " input=[" << input_shape_ss.str()
                      << "] output=[" << output_shape_ss.str() << "]";
}

void EltwiseLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const int64_t count = bottom[0]->count();
  const int num_bottoms = static_cast<int>(bottom.size());

  const char* op_name = "UNKNOWN";
  switch (op_) {
    case PROD: op_name = "PROD"; break;
    case SUM: op_name = "SUM"; break;
    case MAX: op_name = "MAX"; break;
  }

  CAFFE_FFI_LAYER_LOG << "Eltwise Forward: op=" << op_name
                      << " num_bottoms=" << num_bottoms
                      << " count=" << count;

  // TS31-B4 COW promotion: when Eltwise takes a single bottom with coeff == 1,
  // every op degenerates to identity (y = x). Replace the O(n) memcpy with an
  // O(1) refcount share; the COW clone is deferred to the first downstream
  // mutable access. NOTE: the identity check must run BEFORE cpu_mutable_data(),
  // which would otherwise trigger a COW clone on a shared tensor.
  const bool inplace = (bottom[0] == top[0]);
  bool identity = !inplace && num_bottoms == 1 && coeffs_[0] == 1.0f;
  if (identity) {
    top[0]->ShareData(bottom[0]);
    cow_identity_ = true;
    CAFFE_FFI_LOG_INFO() << "[ELTWISE-COW] " << this->name()
                         << " Eltwise forward: IDENTITY (single bottom, coeff=1) -> COW zero-copy"
                         << " op=" << op_name
                         << " count=" << count
                         << " shared_ptr=" << static_cast<const void*>(top[0]->cpu_data())
                         << " bottom_ptr=" << static_cast<const void*>(bottom[0]->cpu_data());
    return;
  }
  cow_identity_ = false;

  float* top_data = top[0]->cpu_mutable_data();

  auto t_start = std::chrono::high_resolution_clock::now();

  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();

  // coeffs值域
  float coeff_min = std::numeric_limits<float>::max();
  float coeff_max = -std::numeric_limits<float>::max();
  for (int j = 0; j < num_bottoms; ++j) {
    coeff_min = std::min(coeff_min, coeffs_[j]);
    coeff_max = std::max(coeff_max, coeffs_[j]);
  }

  switch (op_) {
    case PROD: {
      const float* bottom0_data = bottom[0]->cpu_data();
      #ifdef CAFFE_USE_OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int64_t i = 0; i < count; ++i) {
        top_data[i] = bottom0_data[i] * coeffs_[0];
      }
      for (int j = 1; j < num_bottoms; ++j) {
        const float* bj_data = bottom[j]->cpu_data();
        #ifdef CAFFE_USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t i = 0; i < count; ++i) {
          top_data[i] *= bj_data[i] * coeffs_[j];
        }
      }
      break;
    }
    case SUM: {
      const float* bottom0_data = bottom[0]->cpu_data();
      #ifdef CAFFE_USE_OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int64_t i = 0; i < count; ++i) {
        top_data[i] = bottom0_data[i] * coeffs_[0];
      }
      for (int j = 1; j < num_bottoms; ++j) {
        const float* bj_data = bottom[j]->cpu_data();
        #ifdef CAFFE_USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t i = 0; i < count; ++i) {
          top_data[i] += bj_data[i] * coeffs_[j];
        }
      }
      break;
    }
    case MAX: {
      const float* bottom0_data = bottom[0]->cpu_data();
      #ifdef CAFFE_USE_OPENMP
      #pragma omp parallel for schedule(static)
      #endif
      for (int64_t i = 0; i < count; ++i) {
        top_data[i] = bottom0_data[i] * coeffs_[0];
        max_idx_[i] = 0;
      }
      for (int j = 1; j < num_bottoms; ++j) {
        const float* bj_data = bottom[j]->cpu_data();
        #ifdef CAFFE_USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t i = 0; i < count; ++i) {
          float val = bj_data[i] * coeffs_[j];
          if (val > top_data[i]) {
            top_data[i] = val;
            max_idx_[i] = j;
          }
        }
      }
      break;
    }
    default:
      CAFFE_FFI_THROW(RuntimeError) << "Unknown elementwise operation.";
  }

  // out值域统计
  for (int64_t i = 0; i < count; ++i) {
    out_min = std::min(out_min, top_data[i]);
    out_max = std::max(out_max, top_data[i]);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[ELTWISE-PERF] " << this->name()
                       << " Eltwise forward: op=" << op_name
                       << " num_bottoms=" << num_bottoms
                       << " count=" << count
                       << " coeffs=[" << coeff_min << ", " << coeff_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " time=" << elapsed_us << "us";
}

void EltwiseLayer::Backward_cpu(const std::vector<Blob*>& top,
                                 const std::vector<bool>& propagate_down,
                                 const std::vector<Blob*>& bottom) {
  const int64_t count = bottom[0]->count();
  const float* top_diff = top[0]->cpu_diff();
  const int num_bottoms = static_cast<int>(bottom.size());

  const char* op_name = "UNKNOWN";
  switch (op_) {
    case PROD: op_name = "PROD"; break;
    case SUM: op_name = "SUM"; break;
    case MAX: op_name = "MAX"; break;
  }

  CAFFE_FFI_LAYER_LOG << "Eltwise Backward_cpu: op=" << op_name
                      << " num_bottoms=" << num_bottoms
                      << " count=" << count;

  // Check if any bottom needs gradient
  bool any_propagate = false;
  for (int j = 0; j < num_bottoms; ++j) {
    if (propagate_down[j]) { any_propagate = true; break; }
  }
  if (!any_propagate) {
    CAFFE_FFI_LAYER_LOG << "Eltwise Backward_cpu: no gradients needed, skipping";
    return;
  }

  // TS31-B4 COW promotion: when the forward used the identity short-circuit
  // (single bottom, coeff=1), the backward is a pure identity pass-through
  // (dX = dy). Reuse the O(1) ShareDiff zero-copy instead of an O(n) memcpy.
  if (cow_identity_ && propagate_down[0]) {
    const bool inplace = (bottom[0] == top[0]);
    if (!inplace) {
      bottom[0]->ShareDiff(top[0]);
      CAFFE_FFI_LOG_INFO() << "[ELTWISE-COW] " << this->name()
                           << " Eltwise backward: IDENTITY -> COW zero-copy diff"
                           << " op=" << op_name
                           << " count=" << count
                           << " shared_ptr=" << static_cast<const void*>(bottom[0]->cpu_diff())
                           << " top_ptr=" << static_cast<const void*>(top[0]->cpu_diff());
      return;
    }
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  // Initialize bottom_diff pointers and zero them if needed
  std::vector<float*> bottom_diffs(num_bottoms, nullptr);
  for (int j = 0; j < num_bottoms; ++j) {
    if (propagate_down[j]) {
      bottom_diffs[j] = bottom[j]->cpu_mutable_diff();
      std::memset(bottom_diffs[j], 0, sizeof(float) * count);
    }
  }

  // Value-range tracking for diagnostics
  float dx_min = std::numeric_limits<float>::max();
  float dx_max = -std::numeric_limits<float>::max();

  switch (op_) {
    case SUM: {
      // dX_j = dy * coeffs[j]
      for (int j = 0; j < num_bottoms; ++j) {
        if (!propagate_down[j]) continue;
        float* bj_diff = bottom_diffs[j];
        const float cj = coeffs_[j];
        for (int64_t i = 0; i < count; ++i) {
          float val = top_diff[i] * cj;
          bj_diff[i] = val;
          dx_min = std::min(dx_min, val);
          dx_max = std::max(dx_max, val);
        }
      }
      break;
    }
    case PROD: {
      // dX_j[i] = dy[i] * coeffs[j] * prod_{k≠j}(coeffs[k] * x_k[i])
      // Compute per-element: for each i, compute the product of other terms
      for (int64_t i = 0; i < count; ++i) {
        // Compute product of all terms first
        float prod_all = 1.0f;
        for (int k = 0; k < num_bottoms; ++k) {
          prod_all *= bottom[k]->cpu_data()[i] * coeffs_[k];
        }
        const float dy_val = top_diff[i];
        for (int j = 0; j < num_bottoms; ++j) {
          if (!propagate_down[j]) continue;
          float xj = bottom[j]->cpu_data()[i] * coeffs_[j];
          // dX_j = dy * prod_{k≠j}(...) = dy * prod_all / xj  (when xj ≠ 0)
          float val;
          if (xj != 0.0f) {
            val = dy_val * prod_all / xj;
          } else {
            // When xj=0, directly compute product of other terms
            float prod_others = 1.0f;
            for (int k = 0; k < num_bottoms; ++k) {
              if (k != j) {
                prod_others *= bottom[k]->cpu_data()[i] * coeffs_[k];
              }
            }
            val = dy_val * coeffs_[j] * prod_others;
          }
          bottom_diffs[j][i] = val;
          dx_min = std::min(dx_min, val);
          dx_max = std::max(dx_max, val);
        }
      }
      break;
    }
    case MAX: {
      // dX_j[i] = dy[i] * coeffs[j] if j is winner, else 0
      // Winner indices were recorded in max_idx_ during Forward
      for (int64_t i = 0; i < count; ++i) {
        int winner = max_idx_[i];
        if (propagate_down[winner]) {
          float val = top_diff[i] * coeffs_[winner];
          bottom_diffs[winner][i] = val;
          dx_min = std::min(dx_min, val);
          dx_max = std::max(dx_max, val);
        }
      }
      break;
    }
    default:
      CAFFE_FFI_THROW(RuntimeError) << "Unknown elementwise operation.";
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[ELTWISE-PERF] " << this->name()
                       << " Eltwise backward: op=" << op_name
                       << " num_bottoms=" << num_bottoms
                       << " count=" << count
                       << " dx=[" << dx_min << ", " << dx_max << "]"
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(Eltwise);

}  // namespace caffe_ffi
