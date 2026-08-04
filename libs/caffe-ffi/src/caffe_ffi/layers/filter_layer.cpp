#include "caffe_ffi/layers/filter_layer.hpp"

#include <vector>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"

namespace caffe_ffi {

void FilterLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  CAFFE_FFI_CHECK_VALUE_EQ(top.size(), bottom.size() - 1)
      << "The number of top blobs must equal the number of data bottom blobs "
         "(bottoms minus the selector).";
  first_reshape_ = true;
  CAFFE_FFI_LAYER_LOG << "Filter LayerSetUp: top.size()=" << top.size()
                      << " bottom.size()=" << bottom.size();
}

void FilterLayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  const int selector_index = static_cast<int>(bottom.size()) - 1;
  const int selector_num = static_cast<int>(bottom[selector_index]->shape(0));
  for (int i = 1; i < bottom[selector_index]->num_axes(); ++i) {
    CAFFE_FFI_CHECK_VALUE_EQ(bottom[selector_index]->shape(i), 1)
        << "Selector blob dimensions must be singletons (1), except the first";
  }
  for (int i = 0; i < selector_index; ++i) {
    CAFFE_FFI_CHECK_VALUE_EQ(bottom[selector_index]->shape(0), bottom[i]->shape(0))
        << "Each bottom should have the same 0th dimension as the selector blob";
  }

  const float* bottom_data_selector = bottom[selector_index]->cpu_data();
  indices_to_forward_.clear();
  for (int item_id = 0; item_id < selector_num; ++item_id) {
    const float* tmp_data_selector = bottom_data_selector + item_id;
    if (*tmp_data_selector) {
      indices_to_forward_.push_back(item_id);
    }
  }
  int new_tops_num = static_cast<int>(indices_to_forward_.size());
  if (first_reshape_) {
    new_tops_num = static_cast<int>(bottom[0]->shape(0));
    first_reshape_ = false;
  }
  for (size_t t = 0; t < top.size(); ++t) {
    const int num_axes = bottom[t]->num_axes();
    std::vector<int64_t> shape_top(num_axes);
    shape_top[0] = new_tops_num;
    for (int ts = 1; ts < num_axes; ++ts) {
      shape_top[ts] = bottom[t]->shape(ts);
    }
    top[t]->Reshape(shape_top);
  }
  CAFFE_FFI_LAYER_LOG << "Filter Reshape: indices_to_forward_.size()="
                      << indices_to_forward_.size()
                      << " new_tops_num=" << new_tops_num;
}

void FilterLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  const int new_tops_num = static_cast<int>(indices_to_forward_.size());
  for (size_t t = 0; t < top.size(); ++t) {
    const float* bottom_data = bottom[t]->cpu_data();
    float* top_data = top[t]->cpu_mutable_data();
    const int64_t dim = bottom[t]->count() / bottom[t]->shape(0);
    for (int n = 0; n < new_tops_num; ++n) {
      const int64_t data_offset_top = n * dim;
      const int64_t data_offset_bottom =
          indices_to_forward_[n] * bottom[t]->count(1);
      caffe_copy_fp32(static_cast<size_t>(dim), bottom_data + data_offset_bottom,
                      top_data + data_offset_top);
    }
  }
  CAFFE_FFI_LAYER_LOG << "Filter Forward_cpu: forwarded " << new_tops_num
                      << " items across " << top.size() << " tops";
}

void FilterLayer::Backward_cpu(const std::vector<Blob*>& top,
                               const std::vector<bool>& propagate_down,
                               const std::vector<Blob*>& bottom) {
  if (propagate_down[bottom.size() - 1]) {
    CAFFE_FFI_THROW(RuntimeError)
        << this->type() << "Layer cannot backpropagate to filter index inputs";
  }
  for (size_t i = 0; i < top.size(); ++i) {
    if (propagate_down[i]) {
      const int64_t dim = top[i]->count() / top[i]->shape(0);
      int next_to_backward_offset = 0;
      for (int64_t n = 0; n < bottom[i]->shape(0); ++n) {
        const int64_t data_offset_bottom = n * dim;
        float* bottom_diff = bottom[i]->cpu_mutable_diff();
        if (next_to_backward_offset >= static_cast<int>(indices_to_forward_.size())) {
          caffe_set_fp32(static_cast<size_t>(dim), 0.0f,
                         bottom_diff + data_offset_bottom);
        } else {
          const int batch_offset = indices_to_forward_[next_to_backward_offset];
          if (n != batch_offset) {
            caffe_set_fp32(static_cast<size_t>(dim), 0.0f,
                           bottom_diff + data_offset_bottom);
          } else {
            const int64_t data_offset_top = next_to_backward_offset * dim;
            next_to_backward_offset++;
            caffe_copy_fp32(static_cast<size_t>(dim),
                            top[i]->cpu_diff() + data_offset_top,
                            bottom_diff + data_offset_bottom);
          }
        }
      }
    }
  }
  CAFFE_FFI_LAYER_LOG << "Filter Backward_cpu: completed";
}

REGISTER_LAYER_CLASS(Filter);

}  // namespace caffe_ffi