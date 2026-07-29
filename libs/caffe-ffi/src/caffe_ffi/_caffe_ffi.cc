#include <tvm/ffi/tvm_ffi.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include <google/protobuf/text_format.h>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/net.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/backtrace.hpp"

#include "caffe_ffi/layers/input_layer.hpp"
#include "caffe_ffi/layers/relu_layer.hpp"
#include "caffe_ffi/layers/inner_product_layer.hpp"
#include "caffe_ffi/layers/softmax_layer.hpp"
#include "caffe_ffi/layers/flatten_layer.hpp"
#include "caffe_ffi/layers/conv_layer.hpp"
#include "caffe_ffi/layers/pooling_layer.hpp"
#include "caffe_ffi/layers/batch_norm_layer.hpp"
#include "caffe_ffi/layers/scale_layer.hpp"
#include "caffe_ffi/layers/bias_layer.hpp"
#include "caffe_ffi/layers/softmax_loss_layer.hpp"
#include "caffe_ffi/layers/accuracy_layer.hpp"

#include "caffe_ffi/layers/sigmoid_layer.hpp"
#include "caffe_ffi/layers/tanh_layer.hpp"
#include "caffe_ffi/layers/prelu_layer.hpp"
#include "caffe_ffi/layers/elu_layer.hpp"
#include "caffe_ffi/layers/dropout_layer.hpp"
#include "caffe_ffi/layers/concat_layer.hpp"
#include "caffe_ffi/layers/eltwise_layer.hpp"
#include "caffe_ffi/layers/reshape_layer.hpp"

#include "caffe/proto/caffe.pb.h"

#ifndef CAFFE_FFI_VERSION
#define CAFFE_FFI_VERSION "0.1.0"
#endif

namespace caffe_ffi {

const char* Version() {
  return CAFFE_FFI_VERSION;
}

ObjectPtr<Blob> NewBlob() {
  return make_object<Blob>();
}

ObjectPtr<Blob> NewBlobFromShape(Shape shape) {
  ShapeView sv(shape.data(), shape.size());
  for (size_t i = 0; i < sv.size(); ++i) {
    CAFFE_FFI_CHECK_VALUE_GE(sv[i], 0)
        << "Blob shape dimension " << i << " must be non-negative, got " << sv[i];
  }
  return make_object<Blob>(sv);
}

ObjectPtr<Net> NewNetFromProtoString(const String& proto_text) {
  CAFFE_FFI_CHECK_VALUE(!proto_text.empty()) << "NetParameter proto text must not be empty";
  caffe::NetParameter param = ReadNetParamsFromTextString(static_cast<std::string>(proto_text));
  return make_object<Net>(param);
}

ObjectPtr<Net> NewNetFromFile(const String& filename) {
  CAFFE_FFI_CHECK_VALUE(!filename.empty()) << "Net prototxt filename must not be empty";
  caffe::NetParameter param = ReadNetParamsFromTextFile(static_cast<std::string>(filename));
  return make_object<Net>(param);
}

Array<String> LayerTypeList() {
  auto types = LayerRegistry::LayerTypeList();
  Array<String> result;
  for (const auto& t : types) {
    result.push_back(String(t));
  }
  return result;
}

void SetLogLevel(int level) {
  using caffe_ffi::log::Level;
  if (level < 0) level = 0;
  if (level > 4) level = 4;
  caffe_ffi::log::SetLevel(static_cast<Level>(level));
}

int GetLogLevel() {
  return static_cast<int>(caffe_ffi::log::GetLevel());
}

int64_t TotalAllocatedBytesGlobal() {
  return TotalAllocatedBytes();
}

int64_t LiveBlobCountGlobal() {
  return LiveBlobCount();
}

String GetBacktraceString(int skip_frames, int max_frames) {
  return String(backtrace::GetBacktrace(skip_frames + 1, max_frames));
}

Tensor BlobDataTensor(ObjectPtr<Blob> blob) {
  TVM_FFI_ICHECK(blob != nullptr) << "Blob must not be null";
  CAFFE_FFI_MEM_LOG << "FFI BlobDataTensor blob=" << blob.get()
                    << " returning data_tensor view";
  return blob->data_tensor();
}

Tensor BlobDiffTensor(ObjectPtr<Blob> blob) {
  TVM_FFI_ICHECK(blob != nullptr) << "Blob must not be null";
  CAFFE_FFI_MEM_LOG << "FFI BlobDiffTensor blob=" << blob.get()
                    << " returning diff_tensor view";
  return blob->diff_tensor();
}

void BlobUpdate(ObjectPtr<Blob> blob) {
  TVM_FFI_ICHECK(blob != nullptr) << "Blob must not be null";
  CAFFE_FFI_MEM_LOG << "FFI BlobUpdate blob=" << blob.get();
  blob->Update();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("caffe_ffi.Version", Version, "Get caffe-ffi version string")
      .def("caffe_ffi.NewBlob", NewBlob, "Create an empty Blob")
      .def("caffe_ffi.NewBlobFromShape", NewBlobFromShape, "Create a Blob with specified shape")
      .def("caffe_ffi.NewNetFromProtoString", NewNetFromProtoString, "Create Net from prototxt string")
      .def("caffe_ffi.NewNetFromFile", NewNetFromFile, "Create Net from prototxt file path")
      .def("caffe_ffi.LayerTypeList", LayerTypeList, "List all registered layer type names")
      .def("caffe_ffi.SetLogLevel", SetLogLevel, "Set logging level (0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR)")
      .def("caffe_ffi.GetLogLevel", GetLogLevel, "Get current logging level")
      .def("caffe_ffi.TotalAllocatedBytes", TotalAllocatedBytesGlobal, "Get total bytes allocated across all Blobs")
      .def("caffe_ffi.LiveBlobCount", LiveBlobCountGlobal, "Get number of live Blob objects")
      .def("caffe_ffi.GetBacktrace", GetBacktraceString, "Get stack backtrace string")
      .def("caffe_ffi.BlobDataTensor", BlobDataTensor, "Get Blob's data tensor (zero-copy DLPack)")
      .def("caffe_ffi.BlobDiffTensor", BlobDiffTensor, "Get Blob's diff tensor (zero-copy DLPack)")
      .def("caffe_ffi.BlobUpdate", BlobUpdate, "Update Blob: data -= diff (gradient descent step)");

  refl::ObjectDef<Blob>()
      .def(refl::init<>(), "Create empty Blob")
      .def("shape", static_cast<Shape (Blob::*)() const>(&Blob::shape), "Get Blob shape as TVM FFI Shape")
      .def("shape_at", static_cast<int64_t (Blob::*)(int) const>(&Blob::shape), "Get dimension size at axis")
      .def("num_axes", &Blob::num_axes, "Get number of dimensions (axes)")
      .def("count", static_cast<int64_t (Blob::*)() const>(&Blob::count), "Get total number of elements")
      .def("count_from", static_cast<int64_t (Blob::*)(int) const>(&Blob::count), "Count elements from start_axis to end")
      .def("count_range", static_cast<int64_t (Blob::*)(int, int) const>(&Blob::count), "Count elements in [start_axis, end_axis)")
      .def("canonical_axis_index", &Blob::CanonicalAxisIndex, "Convert negative axis to canonical index")
      .def("num", &Blob::num, "Legacy API: get batch size (dimension 0)")
      .def("channels", &Blob::channels, "Legacy API: get channel count (dimension 1)")
      .def("height", &Blob::height, "Legacy API: get spatial height (dimension 2)")
      .def("width", &Blob::width, "Legacy API: get spatial width (dimension 3)")
      .def("Reshape", static_cast<void (Blob::*)(Shape)>(&Blob::Reshape), "Reshape Blob (reallocate if needed)")
      .def("get_data", &Blob::get_data, "Get data as Array<float> (copies data)")
      .def("set_data", &Blob::set_data, "Set data from Tensor (supports numpy zero-copy via DLPack)")
      .def("get_diff", &Blob::get_diff, "Get diff as Array<float> (copies data)")
      .def("set_diff", &Blob::set_diff, "Set diff from Tensor (supports numpy zero-copy via DLPack)")
      .def("data_tensor", &Blob::data_tensor, "Get data tensor (zero-copy DLPack interop)")
      .def("diff_tensor", &Blob::diff_tensor, "Get diff tensor (zero-copy DLPack interop)")
      .def("Update", &Blob::Update, "Update data: data -= diff (gradient step)")
      .def("name", &Blob::name, "Get Blob name")
      .def("set_name", &Blob::set_name, "Set Blob name")
      .def("id", &Blob::id, "Get unique Blob ID (debugging)")
      .def("construction_backtrace", &Blob::construction_backtrace, "Get construction stack backtrace (debug)");

  refl::ObjectDef<Layer>()
      .def("type", &Layer::type, "Get layer type string (e.g., 'ReLU', 'InnerProduct')")
      .def("name", &Layer::name, "Get layer name")
      .def("blobs_array", &Layer::blobs_array, "Get parameter blobs (weights/biases) as Array");

  refl::ObjectDef<Net>()
      .def("name", &Net::name, "Get network name")
      .def("Forward", &Net::Forward, "Run forward pass (returns Map of output name -> Blob)")
      .def("ForwardFromTo", &Net::ForwardFromTo, "Run forward pass from start to end layer (returns total loss)")
      .def("CopyTrainedLayersFrom", static_cast<void (Net::*)(const std::string&)>(&Net::CopyTrainedLayersFrom), "Load trained weights from .caffemodel file")
      .def("blobs_array", &Net::blobs_array, "Get all blobs as Array")
      .def("layers_array", &Net::layers_array, "Get all layers as Array")
      .def("input_blobs_array", &Net::input_blobs_array, "Get input blobs as Array")
      .def("output_blobs_array", &Net::output_blobs_array, "Get output blobs as Array")
      .def("blob_by_name", &Net::blob_by_name, "Look up blob by name (throws KeyError if not found)")
      .def("layer_by_name", &Net::layer_by_name, "Look up layer by name (throws KeyError if not found)")
      .def("has_blob", &Net::has_blob, "Check if blob exists by name")
      .def("has_layer", &Net::has_layer, "Check if layer exists by name")
      .def("num_inputs", &Net::num_inputs, "Get number of input blobs")
      .def("num_outputs", &Net::num_outputs, "Get number of output blobs")
      .def("input_blob_names", &Net::input_blob_names_array, "Get input blob names as Array<String>")
      .def("output_blob_names", &Net::output_blob_names_array, "Get output blob names as Array<String>")
      .def("blob_names", &Net::blob_names_array, "Get all blob names as Array<String>")
      .def("layer_names", &Net::layer_names_array, "Get all layer names as Array<String>");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(Version, Version)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(NewBlob, NewBlob)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(NewBlobFromShape, NewBlobFromShape)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(NewNetFromProtoString, NewNetFromProtoString)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(NewNetFromFile, NewNetFromFile)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(LayerTypeList, LayerTypeList)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(SetLogLevel, SetLogLevel)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(GetLogLevel, GetLogLevel)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(TotalAllocatedBytes, TotalAllocatedBytesGlobal)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(LiveBlobCount, LiveBlobCountGlobal)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(GetBacktrace, GetBacktraceString)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(BlobDataTensor, BlobDataTensor)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(BlobDiffTensor, BlobDiffTensor)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(BlobUpdate, BlobUpdate)

}  // namespace caffe_ffi
