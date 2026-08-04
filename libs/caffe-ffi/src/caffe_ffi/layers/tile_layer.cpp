#include "caffe_ffi/layers/tile_layer.hpp"

#include <cstring>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void TileLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) {
  axis_ = this->layer_param_.tile_param().axis();
  tiles_ = this->layer_param_.tile_param().tiles();

  CAFFE_FFI_LAYER_LOG << "Tile LayerSetUp: axis=" << axis_ << " tiles=" << tiles_;
}

void TileLayer::Reshape(const std::vector<Blob*>& bottom,
                        const std::vector<Blob*>& top) {
  axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.tile_param().axis());
  tiles_ = this->layer_param_.tile_param().tiles();

  std::vector<int64_t> top_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  top_shape[axis_] = bottom[0]->shape(axis_) * tiles_;
  top[0]->Reshape(top_shape);

  outer_dim_ = bottom[0]->count(0, axis_);
  inner_dim_ = bottom[0]->count(axis_);

  CAFFE_FFI_LAYER_LOG << "Tile Reshape: axis=" << axis_ << " tiles=" << tiles_
                      << " outer_dim_=" << outer_dim_ << " inner_dim_=" << inner_dim_;
}

void TileLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();

  for (int64_t i = 0; i < outer_dim_; ++i) {
    for (int t = 0; t < tiles_; ++t) {
      std::memcpy(top_data, bottom_data, sizeof(float) * inner_dim_);
      top_data += inner_dim_;
    }
    bottom_data += inner_dim_;
  }

  CAFFE_FFI_LAYER_LOG << "Tile Forward: outer_dim_=" << outer_dim_
                      << " inner_dim_=" << inner_dim_ << " tiles=" << tiles_;
}

void TileLayer::Backward_cpu(const std::vector<Blob*>& top,
                             const std::vector<bool>& propagate_down,
                             const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Tile Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();

  for (int64_t i = 0; i < outer_dim_; ++i) {
    std::memcpy(bottom_diff, top_diff, sizeof(float) * inner_dim_);
    top_diff += inner_dim_;
    for (int t = 1; t < tiles_; ++t) {
      for (int64_t j = 0; j < inner_dim_; ++j) {
        bottom_diff[j] += top_diff[j];
      }
      top_diff += inner_dim_;
    }
    bottom_diff += inner_dim_;
  }

  CAFFE_FFI_LAYER_LOG << "Tile Backward: outer_dim_=" << outer_dim_
                      << " inner_dim_=" << inner_dim_ << " tiles=" << tiles_;
}

REGISTER_LAYER_CLASS(Tile);

}  // namespace caffe_ffi