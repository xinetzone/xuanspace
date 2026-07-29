#include "caffe_ffi/blob.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/backtrace.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

std::atomic<int64_t> g_total_allocated_bytes{0};

namespace {

std::atomic<int64_t> g_live_blob_count{0};
std::atomic<int64_t> g_next_blob_id{1};

std::string ShapeToString(ShapeView shape) {
  std::ostringstream oss;
  oss << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << ",";
    oss << shape[i];
  }
  oss << ")";
  return oss.str();
}

int64_t TensorNBytes(const Tensor& t) {
  if (!t.defined()) return 0;
  return t.numel() * static_cast<int64_t>(t.dtype().bits / 8);
}

std::string PtrToString(const void* p) {
  std::ostringstream oss;
  oss << p;
  return oss.str();
}

std::string FormatBytes(int64_t bytes) {
  std::ostringstream oss;
  if (bytes < 0) {
    oss << "-";
    bytes = -bytes;
  }
  if (bytes >= 1024 * 1024) {
    oss << (bytes / (1024.0 * 1024.0)) << " MB";
  } else if (bytes >= 1024) {
    oss << (bytes / 1024.0) << " KB";
  } else {
    oss << bytes << " B";
  }
  return oss.str();
}

}  // namespace

Blob::Blob() : id_(g_next_blob_id.fetch_add(1, std::memory_order_relaxed)) {
  g_live_blob_count.fetch_add(1, std::memory_order_relaxed);
  construct_bt_ = backtrace::GetBacktrace(3);
  CAFFE_FFI_BLOB_LOG << "[MEM-LIFECYCLE] Blob#" << id_ << " constructed (default) this=" << this
                     << " live_blobs=" << g_live_blob_count.load(std::memory_order_relaxed);
  Reshape(std::vector<int64_t>{0});
}

Blob::Blob(ShapeView shape) : id_(g_next_blob_id.fetch_add(1, std::memory_order_relaxed)) {
  g_live_blob_count.fetch_add(1, std::memory_order_relaxed);
  construct_bt_ = backtrace::GetBacktrace(3);
  CAFFE_FFI_BLOB_LOG << "[MEM-LIFECYCLE] Blob#" << id_ << " constructed (ShapeView) this=" << this
                     << " shape=" << ShapeToString(shape)
                     << " live_blobs=" << g_live_blob_count.load(std::memory_order_relaxed);
  Reshape(shape);
}

Blob::Blob(const std::vector<int64_t>& shape)
    : id_(g_next_blob_id.fetch_add(1, std::memory_order_relaxed)) {
  g_live_blob_count.fetch_add(1, std::memory_order_relaxed);
  construct_bt_ = backtrace::GetBacktrace(3);
  CAFFE_FFI_BLOB_LOG << "[MEM-LIFECYCLE] Blob#" << id_ << " constructed (vector) this=" << this
                     << " shape=" << ShapeToString(ShapeView(shape.data(), shape.size()))
                     << " live_blobs=" << g_live_blob_count.load(std::memory_order_relaxed);
  Reshape(ShapeView(shape.data(), shape.size()));
}

Blob::~Blob() {
  int64_t data_nbytes = TensorNBytes(data_tensor_);
  int64_t diff_nbytes = TensorNBytes(diff_tensor_);
  int64_t total_freed = data_nbytes + diff_nbytes;

  int64_t live_before = g_live_blob_count.fetch_sub(1, std::memory_order_relaxed);
  int64_t live_after = live_before - 1;

  CAFFE_FFI_MEM_LOG << "[MEM-FREE] Blob#" << id_ << " this=" << this
                    << " shape=" << ShapeToString(ShapeView(data_tensor_.shape().data(),
                                                            static_cast<size_t>(data_tensor_.ndim())))
                    << " data_ptr=" << PtrToString(data_tensor_.data_ptr())
                    << " diff_ptr=" << PtrToString(diff_tensor_.data_ptr())
                    << " freed=" << total_freed << "B (" << FormatBytes(total_freed) << ")"
                    << " live_blobs=" << live_after;

  data_tensor_ = Tensor();
  diff_tensor_ = Tensor();

  CAFFE_FFI_LOG_TRACE() << "[MEM-LIFECYCLE] Blob#" << id_
                        << " construction backtrace:\n" << construct_bt_;

  CAFFE_FFI_MEM_LOG << "[MEM-LIFECYCLE] Blob#" << id_
                    << " destroyed, global_total=" << g_total_allocated_bytes.load(std::memory_order_relaxed) << "B"
                    << " live_blobs=" << live_after << " (was " << live_before << ")";
}

Tensor Blob::data_tensor() const {
  CAFFE_FFI_TENSOR_LOG << "data_tensor() Blob#" << id_ << " this=" << this
                       << " ptr=" << PtrToString(data_tensor_.data_ptr())
                       << " shape=" << ShapeToString(ShapeView(data_tensor_.shape().data(),
                                                               static_cast<size_t>(data_tensor_.ndim())))
                       << " numel=" << data_tensor_.numel()
                       << " nbytes=" << TensorNBytes(data_tensor_)
                       << " dtype=" << DTypeCodeToString(data_tensor_.dtype().code)
                       << static_cast<int>(data_tensor_.dtype().bits)
                       << " device_type=" << static_cast<int>(data_tensor_.device().device_type);
  return data_tensor_;
}

Tensor Blob::diff_tensor() const {
  CAFFE_FFI_TENSOR_LOG << "diff_tensor() Blob#" << id_ << " this=" << this
                       << " ptr=" << PtrToString(diff_tensor_.data_ptr())
                       << " shape=" << ShapeToString(ShapeView(diff_tensor_.shape().data(),
                                                               static_cast<size_t>(diff_tensor_.ndim())))
                       << " numel=" << diff_tensor_.numel()
                       << " nbytes=" << TensorNBytes(diff_tensor_)
                       << " dtype=" << DTypeCodeToString(diff_tensor_.dtype().code)
                       << static_cast<int>(diff_tensor_.dtype().bits)
                       << " device_type=" << static_cast<int>(diff_tensor_.device().device_type);
  return diff_tensor_;
}

void Blob::Reshape(ShapeView shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    CAFFE_FFI_CHECK_VALUE_GE(shape[i], 0)
        << "Blob#" << id_ << " Reshape: dimension " << i << " is negative (" << shape[i] << ")";
  }
  bool shape_changed = !data_tensor_.defined() || (shape.size() != static_cast<size_t>(data_tensor_.ndim()));
  if (!shape_changed) {
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] != data_tensor_.size(static_cast<int>(i))) {
        shape_changed = true;
        break;
      }
    }
  }
  int64_t new_count = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    new_count *= shape[i];
  }
  int64_t old_count = data_tensor_.defined() ? data_tensor_.numel() : 0;
  const void* old_data_ptr = data_tensor_.defined() ? data_tensor_.data_ptr() : nullptr;
  const void* old_diff_ptr = diff_tensor_.defined() ? diff_tensor_.data_ptr() : nullptr;
  int64_t old_nbytes = TensorNBytes(data_tensor_) + TensorNBytes(diff_tensor_);

  if (shape_changed || !data_tensor_.defined()) {
    int64_t new_total_nbytes = new_count * sizeof(float) * 2;
    int64_t net_delta = new_total_nbytes - old_nbytes;

    CAFFE_FFI_MEM_LOG << "[MEM-RESIZE] Blob#" << id_ << " this=" << this
                      << " shape=" << ShapeToString(shape)
                      << " old_count=" << old_count << " new_count=" << new_count
                      << " old_data_ptr=" << PtrToString(old_data_ptr)
                      << " old_diff_ptr=" << PtrToString(old_diff_ptr)
                      << " old_nbytes=" << old_nbytes << "B (" << FormatBytes(old_nbytes) << ")"
                      << " new_nbytes=" << new_total_nbytes << "B (" << FormatBytes(new_total_nbytes) << ")"
                      << " net_delta=" << (net_delta >= 0 ? "+" : "") << net_delta << "B";

    int64_t global_before = g_total_allocated_bytes.load(std::memory_order_relaxed);

    data_tensor_ = NewCPUTensor(shape);
    diff_tensor_ = NewCPUTensor(shape);

    int64_t global_after = g_total_allocated_bytes.load(std::memory_order_relaxed);

    CAFFE_FFI_MEM_LOG << "[MEM-RESIZE] Blob#" << id_
                      << " new_data_ptr=" << PtrToString(data_tensor_.data_ptr())
                      << " new_diff_ptr=" << PtrToString(diff_tensor_.data_ptr())
                      << " global_delta=" << (net_delta >= 0 ? "+" : "") << net_delta << "B"
                      << " global_before=" << global_before << "B (" << FormatBytes(global_before) << ")"
                      << " global_after=" << global_after << "B (" << FormatBytes(global_after) << ")"
                      << " live_blobs=" << g_live_blob_count.load(std::memory_order_relaxed);
  } else {
    CAFFE_FFI_TENSOR_LOG << "Reshape: Blob#" << id_ << " shape unchanged " << ShapeToString(shape)
                         << " (count=" << new_count << "), skipping reallocation"
                         << " data_ptr=" << PtrToString(data_tensor_.data_ptr())
                         << " diff_ptr=" << PtrToString(diff_tensor_.data_ptr());
  }
}

void Blob::Reshape(const std::vector<int64_t>& shape) {
  Reshape(ShapeView(shape.data(), shape.size()));
}

void Blob::Reshape(Shape shape) {
  Reshape(ShapeView(shape.data(), shape.size()));
}

void Blob::Reshape(const caffe::BlobShape& shape) {
  std::vector<int64_t> dims;
  for (int i = 0; i < shape.dim_size(); ++i) {
    dims.push_back(shape.dim(i));
  }
  Reshape(dims);
}

void Blob::ReshapeLike(const Blob& other) {
  std::vector<int64_t> dims;
  for (int i = 0; i < other.num_axes(); ++i) {
    dims.push_back(other.shape(i));
  }
  Reshape(dims);
}

int64_t Blob::LegacyShape(int index) const {
  if (index >= num_axes()) {
    return 1;
  }
  return shape(index);
}

void Blob::FromProto(const caffe::BlobProto& proto, bool reshape) {
  CAFFE_FFI_CONTAINER_LOG << "FromProto: Blob#" << id_ << " reshape=" << reshape
                           << " proto.data_size=" << proto.data_size()
                           << " proto.double_data_size=" << proto.double_data_size()
                           << " proto.diff_size=" << proto.diff_size();
  if (reshape) {
    std::vector<int64_t> shape;
    if (proto.has_shape()) {
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape.push_back(proto.shape().dim(i));
      }
    } else {
      if (proto.num() > 0) shape.push_back(proto.num());
      if (proto.channels() > 0) shape.push_back(proto.channels());
      if (proto.height() > 0) shape.push_back(proto.height());
      if (proto.width() > 0) shape.push_back(proto.width());
      if (shape.empty()) shape.push_back(0);
    }
    CAFFE_FFI_TENSOR_LOG << "FromProto: Blob#" << id_ << " reshaping to " << ShapeToString(ShapeView(shape.data(), shape.size()));
    Reshape(shape);
  }
  float* data_ptr = cpu_data();
  const int data_count = proto.data_size();
  const int double_data_count = proto.double_data_size();
  if (data_count > 0) {
    CAFFE_FFI_CHECK_RUNTIME_EQ(data_count, count())
        << "Incorrect data size for Blob: expected " << count() << ", got " << data_count;
    CAFFE_FFI_CONTAINER_LOG << "FromProto: Blob#" << id_ << " copying " << data_count << " float data elements to " << data_ptr;
    std::copy(proto.data().begin(), proto.data().end(), data_ptr);
  } else if (double_data_count > 0) {
    CAFFE_FFI_CHECK_RUNTIME_EQ(double_data_count, count())
        << "Incorrect double_data size for Blob: expected " << count() << ", got " << double_data_count;
    CAFFE_FFI_CONTAINER_LOG << "FromProto: Blob#" << id_ << " converting " << double_data_count << " double→float data elements";
    for (int i = 0; i < double_data_count; ++i) {
      data_ptr[i] = static_cast<float>(proto.double_data(i));
    }
  }
  float* diff_ptr = cpu_diff();
  const int diff_count = proto.diff_size();
  const int double_diff_count = proto.double_diff_size();
  if (diff_count > 0) {
    CAFFE_FFI_CHECK_RUNTIME_EQ(diff_count, count())
        << "Incorrect diff size for Blob: expected " << count() << ", got " << diff_count;
    CAFFE_FFI_CONTAINER_LOG << "FromProto: Blob#" << id_ << " copying " << diff_count << " float diff elements to " << diff_ptr;
    std::copy(proto.diff().begin(), proto.diff().end(), diff_ptr);
  } else if (double_diff_count > 0) {
    CAFFE_FFI_CHECK_RUNTIME_EQ(double_diff_count, count())
        << "Incorrect double_diff size for Blob: expected " << count() << ", got " << double_diff_count;
    CAFFE_FFI_CONTAINER_LOG << "FromProto: Blob#" << id_ << " converting " << double_diff_count << " double→float diff elements";
    for (int i = 0; i < double_diff_count; ++i) {
      diff_ptr[i] = static_cast<float>(proto.double_diff(i));
    }
  } else {
    CAFFE_FFI_CONTAINER_LOG << "FromProto: Blob#" << id_ << " zeroing diff tensor (" << count() << " elements at " << diff_ptr << ")";
    caffe_set_fp32(static_cast<size_t>(count()), 0.0f, diff_ptr);
  }
  CAFFE_FFI_BLOB_LOG << "FromProto: Blob#" << id_ << " completed, data_ptr=" << data_ptr << " diff_ptr=" << diff_ptr;
}

void Blob::ToProto(caffe::BlobProto* proto) const {
  proto->Clear();
  auto* shape_proto = proto->mutable_shape();
  shape_proto->clear_dim();
  for (int i = 0; i < num_axes(); ++i) {
    shape_proto->add_dim(shape(i));
  }
  proto->clear_data();
  const float* data_ptr = cpu_data();
  for (int64_t i = 0; i < count(); ++i) {
    proto->add_data(data_ptr[i]);
  }
  proto->clear_diff();
  const float* diff_ptr = cpu_diff();
  for (int64_t i = 0; i < count(); ++i) {
    proto->add_diff(diff_ptr[i]);
  }
}

void Blob::Update() {
  CAFFE_FFI_TENSOR_LOG << "Update() Blob#" << id_ << " this=" << this
                       << " data_ptr=" << PtrToString(cpu_data())
                       << " diff_ptr=" << PtrToString(cpu_diff())
                       << " count=" << count()
                       << " operation: data -= diff";
  caffe_cpu_axpby_fp32(static_cast<size_t>(count()), -1.0f, cpu_diff(), 1.0f, cpu_data());
}

Array<float> Blob::get_data() const {
  Array<float> result;
  result.reserve(count());
  const float* ptr = cpu_data();
  CAFFE_FFI_CONTAINER_LOG << "get_data: Blob#" << id_ << " copying " << count() << " elements from " << ptr << " to Array<float>";
  for (int64_t i = 0; i < count(); ++i) {
    result.push_back(ptr[i]);
  }
  CAFFE_FFI_CONTAINER_LOG << "get_data: Blob#" << id_ << " Array<float> size=" << result.size() << " created";
  return result;
}

void Blob::set_data(Tensor data) {
  // Support DLPack zero-copy interop with numpy/PyTorch/etc.
  CAFFE_FFI_CHECK_TYPE(data.defined()) << "Cannot set_data from undefined Tensor (Blob#" << id_ << ")";
  CAFFE_FFI_CHECK_VALUE_EQ(data.ndim(), num_axes())
      << "Tensor ndim mismatch for Blob#" << id_ << ": expected " << num_axes() << ", got " << data.ndim();
  for (int i = 0; i < data.ndim(); ++i) {
    CAFFE_FFI_CHECK_VALUE_EQ(data.size(i), shape(i))
        << "Tensor shape mismatch at axis " << i << " for Blob#" << id_
        << ": expected " << shape(i) << ", got " << data.size(i);
  }
  CAFFE_FFI_CHECK_TYPE(data.dtype().code == kDLFloat && data.dtype().bits == 32)
      << "set_data expects float32 Tensor for Blob#" << id_
      << ", got dtype code=" << static_cast<int>(data.dtype().code) << " bits=" << data.dtype().bits;

  float* dst = cpu_data();
  const float* src = static_cast<const float*>(data.data_ptr());
  int64_t nbytes = count() * sizeof(float);
  CAFFE_FFI_CONTAINER_LOG << "set_data(Tensor): Blob#" << id_ << " memcpy " << count()
                          << " elements (" << nbytes << "B) from " << src << " to " << dst;
  std::memcpy(dst, src, nbytes);
}

Array<float> Blob::get_diff() const {
  Array<float> result;
  result.reserve(count());
  const float* ptr = cpu_diff();
  CAFFE_FFI_CONTAINER_LOG << "get_diff: Blob#" << id_ << " copying " << count() << " elements from " << ptr << " to Array<float>";
  for (int64_t i = 0; i < count(); ++i) {
    result.push_back(ptr[i]);
  }
  return result;
}

void Blob::set_diff(Tensor diff) {
  CAFFE_FFI_CHECK_TYPE(diff.defined()) << "Cannot set_diff from undefined Tensor (Blob#" << id_ << ")";
  CAFFE_FFI_CHECK_VALUE_EQ(diff.ndim(), num_axes())
      << "Tensor ndim mismatch for Blob#" << id_ << " diff: expected " << num_axes() << ", got " << diff.ndim();
  for (int i = 0; i < diff.ndim(); ++i) {
    CAFFE_FFI_CHECK_VALUE_EQ(diff.size(i), shape(i))
        << "Tensor shape mismatch at axis " << i << " for Blob#" << id_ << " diff"
        << ": expected " << shape(i) << ", got " << diff.size(i);
  }
  CAFFE_FFI_CHECK_TYPE(diff.dtype().code == kDLFloat && diff.dtype().bits == 32)
      << "set_diff expects float32 Tensor for Blob#" << id_
      << ", got dtype code=" << static_cast<int>(diff.dtype().code) << " bits=" << diff.dtype().bits;

  float* dst = cpu_diff();
  const float* src = static_cast<const float*>(diff.data_ptr());
  int64_t nbytes = count() * sizeof(float);
  CAFFE_FFI_CONTAINER_LOG << "set_diff(Tensor): Blob#" << id_ << " memcpy " << count()
                          << " elements (" << nbytes << "B) from " << src << " to " << dst;
  std::memcpy(dst, src, nbytes);
}

int64_t TotalAllocatedBytes() {
  int64_t val = g_total_allocated_bytes.load(std::memory_order_relaxed);
  CAFFE_FFI_MEM_LOG << "[MEM-QUERY] TotalAllocatedBytes() = " << val << "B (" << FormatBytes(val) << ")"
                    << " live_blobs=" << g_live_blob_count.load(std::memory_order_relaxed);
  return val;
}

int64_t LiveBlobCount() {
  int64_t val = g_live_blob_count.load(std::memory_order_relaxed);
  CAFFE_FFI_MEM_LOG << "[MEM-QUERY] LiveBlobCount() = " << val
                    << " total_allocated=" << g_total_allocated_bytes.load(std::memory_order_relaxed) << "B";
  return val;
}

}  // namespace caffe_ffi
