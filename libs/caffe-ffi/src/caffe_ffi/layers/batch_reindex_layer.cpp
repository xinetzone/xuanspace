#include "caffe_ffi/layers/batch_reindex_layer.hpp"

#include <vector>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"

namespace caffe_ffi {

void BatchReindexLayer::Reshape(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[1]->num_axes(), 1)
      << "The index blob must be 1-dimensional.";
  std::vector<int64_t> newshape;
  newshape.push_back(bottom[1]->shape(0));
  for (int i = 1; i < bottom[0]->num_axes(); ++i) {
    newshape.push_back(bottom[0]->shape(i));
  }
  top[0]->Reshape(newshape);
  CAFFE_FFI_LAYER_LOG << "BatchReindex Reshape: output batch=" << newshape[0]
                      << " (from index blob count)";
}

void BatchReindexLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                    const std::vector<Blob*>& top) {
  const int initial_num = static_cast<int>(bottom[0]->shape(0));
  const int final_num = static_cast<int>(bottom[1]->count());
  const float* ridx_data = bottom[1]->cpu_data();
  for (int i = 0; i < final_num; ++i) {
    CAFFE_FFI_CHECK_VALUE_GE(ridx_data[i], 0)
        << "Index specified for reindex layer was negative.";
    CAFFE_FFI_CHECK_VALUE_LT(ridx_data[i], initial_num)
        << "Index specified for reindex layer was greater than batch size.";
  }
  if (top[0]->count() == 0) {
    return;
  }
  const int64_t inner_dim = bottom[0]->count() / initial_num;
  const float* in = bottom[0]->cpu_data();
  const float* permut = bottom[1]->cpu_data();
  float* out = top[0]->cpu_mutable_data();
  for (int64_t index = 0; index < top[0]->count(); ++index) {
    const int64_t n = index / inner_dim;
    const int in_n = static_cast<int>(permut[n]);
    out[index] = in[in_n * inner_dim + index % inner_dim];
  }
  CAFFE_FFI_LAYER_LOG << "BatchReindex Forward_cpu: initial_num=" << initial_num
                      << " final_num=" << final_num << " inner_dim=" << inner_dim;
}

void BatchReindexLayer::Backward_cpu(const std::vector<Blob*>& top,
                                     const std::vector<bool>& propagate_down,
                                     const std::vector<Blob*>& bottom) {
  CAFFE_FFI_CHECK_VALUE(!propagate_down[1])
      << "Cannot backprop to index.";
  if (!propagate_down[0]) {
    return;
  }
  const int initial_num = static_cast<int>(bottom[0]->shape(0));
  const int64_t inner_dim = bottom[0]->count() / initial_num;
  float* bot_diff = bottom[0]->cpu_mutable_diff();
  const float* permut = bottom[1]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  caffe_set_fp32(static_cast<size_t>(bottom[0]->count()), 0.0f, bot_diff);
  for (int64_t index = 0; index < top[0]->count(); ++index) {
    const int64_t n = index / inner_dim;
    const int in_n = static_cast<int>(permut[n]);
    bot_diff[in_n * inner_dim + index % inner_dim] += top_diff[index];
  }
  CAFFE_FFI_LAYER_LOG << "BatchReindex Backward_cpu: scatter-add completed";
}

REGISTER_LAYER_CLASS(BatchReindex);

}  // namespace caffe_ffi