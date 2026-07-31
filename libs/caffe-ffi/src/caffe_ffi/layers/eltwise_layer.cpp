#include "caffe_ffi/layers/eltwise_layer.hpp"

#include <algorithm>
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
  float* top_data = top[0]->cpu_mutable_data();
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

  switch (op_) {
    case PROD: {
      const float* bottom0_data = bottom[0]->cpu_data();
      for (int64_t i = 0; i < count; ++i) {
        top_data[i] = bottom0_data[i] * coeffs_[0];
      }
      for (int j = 1; j < num_bottoms; ++j) {
        const float* bj_data = bottom[j]->cpu_data();
        for (int64_t i = 0; i < count; ++i) {
          top_data[i] *= bj_data[i] * coeffs_[j];
        }
      }
      break;
    }
    case SUM: {
      const float* bottom0_data = bottom[0]->cpu_data();
      for (int64_t i = 0; i < count; ++i) {
        top_data[i] = bottom0_data[i] * coeffs_[0];
      }
      for (int j = 1; j < num_bottoms; ++j) {
        const float* bj_data = bottom[j]->cpu_data();
        for (int64_t i = 0; i < count; ++i) {
          top_data[i] += bj_data[i] * coeffs_[j];
        }
      }
      break;
    }
    case MAX: {
      const float* bottom0_data = bottom[0]->cpu_data();
      for (int64_t i = 0; i < count; ++i) {
        top_data[i] = bottom0_data[i] * coeffs_[0];
      }
      for (int j = 1; j < num_bottoms; ++j) {
        const float* bj_data = bottom[j]->cpu_data();
        for (int64_t i = 0; i < count; ++i) {
          top_data[i] = std::max(top_data[i], bj_data[i] * coeffs_[j]);
        }
      }
      break;
    }
    default:
      CAFFE_FFI_THROW(RuntimeError) << "Unknown elementwise operation.";
  }
}

REGISTER_LAYER_CLASS(Eltwise);

}  // namespace caffe_ffi
