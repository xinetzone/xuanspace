#ifndef CAFFE_FFI_NET_HPP_
#define CAFFE_FFI_NET_HPP_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Neural network container that manages layers, blobs, and forward propagation.
 *
 * Net constructs a directed acyclic graph (DAG) of layers from a NetParameter protobuf
 * definition (typically loaded from a .prototxt file). It handles blob name resolution,
 * topological ordering, memory allocation for intermediate blobs, and forward propagation
 * through all layers.
 *
 * After construction, call Forward() with input data to compute outputs. Trained weights
 * can be loaded via CopyTrainedLayersFrom() from a .caffemodel file.
 */
class Net : public Object {
 public:
  static constexpr bool _type_mutable = true;

  /** @brief Construct a network from a NetParameter protobuf message. */
  explicit Net(const caffe::NetParameter& param);
  /** @brief Construct a network from a .prototxt file path. */
  explicit Net(const std::string& param_file);
  virtual ~Net() = default;

  /** @brief Initialize the network from a NetParameter (called by constructors). */
  void Init(const caffe::NetParameter& param);

  /** @brief Load trained weights from a .caffemodel file. */
  void CopyTrainedLayersFrom(const std::string& trained_filename);
  /** @brief Load trained weights from a NetParameter protobuf message. */
  void CopyTrainedLayersFrom(const caffe::NetParameter& trained_net_param);

  /**
   * @brief Run forward pass through all layers.
   * @param inputs Map from input blob name to data as Tensor (numpy zero-copy interop via DLPack).
   * @return Map from output blob name to output Blob (zero-copy reference).
   */
  Map<String, ObjectPtr<Blob>> Forward(
      const Map<String, Tensor>& inputs = {});

  /**
   * @brief Run forward pass from layer start to layer end (inclusive).
   * @param start Index of first layer to run.
   * @param end Index of last layer to run.
   * @return Total loss from all loss layers in the range.
   */
  float ForwardFromTo(int start, int end);

  /** @brief Get the network name. */
  const std::string& name() const { return name_; }
  /** @brief Get ordered list of all layer names. */
  const std::vector<std::string>& layer_names() const { return layer_names_; }
  /** @brief Get ordered list of all blob names. */
  const std::vector<std::string>& blob_names() const { return blob_names_; }

  /** @brief Get layer names as a TVM FFI Array for Python interop. */
  Array<String> layer_names_array() const;
  /** @brief Get blob names as a TVM FFI Array for Python interop. */
  Array<String> blob_names_array() const;
  /** @brief Get input blob names as a TVM FFI Array. */
  Array<String> input_blob_names_array() const;
  /** @brief Get output blob names as a TVM FFI Array. */
  Array<String> output_blob_names_array() const;

  /** @brief Get all blobs as a TVM FFI Array for Python interop. */
  Array<ObjectPtr<Blob>> blobs_array() const;
  /** @brief Get all layers as a TVM FFI Array for Python interop. */
  Array<ObjectPtr<Layer>> layers_array() const;
  /** @brief Get input blobs as a TVM FFI Array. */
  Array<ObjectPtr<Blob>> input_blobs_array() const;
  /** @brief Get output blobs as a TVM FFI Array. */
  Array<ObjectPtr<Blob>> output_blobs_array() const;

  /**
   * @brief Look up a blob by name.
   * @throws KeyError if blob_name is not found.
   */
  ObjectPtr<Blob> blob_by_name(const std::string& blob_name) const;
  /**
   * @brief Look up a layer by name.
   * @throws KeyError if layer_name is not found.
   */
  ObjectPtr<Layer> layer_by_name(const std::string& layer_name) const;

  /** @brief Check if a blob with the given name exists. */
  bool has_blob(const std::string& blob_name) const;
  /** @brief Check if a layer with the given name exists. */
  bool has_layer(const std::string& layer_name) const;

  /** @brief Get number of input blobs. */
  int num_inputs() const { return static_cast<int>(net_input_blobs_.size()); }
  /** @brief Get number of output blobs. */
  int num_outputs() const { return static_cast<int>(net_output_blobs_.size()); }
  /** @brief Get raw pointers to input blobs. */
  const std::vector<Blob*>& input_blobs() const { return net_input_blobs_; }
  /** @brief Get raw pointers to output blobs. */
  const std::vector<Blob*>& output_blobs() const { return net_output_blobs_; }
  /** @brief Get input blob names. */
  const std::vector<std::string>& input_blob_names() const { return net_input_blob_names_; }
  /** @brief Get output blob names. */
  const std::vector<std::string>& output_blob_names() const { return net_output_blob_names_; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "caffe_ffi.Net", Net, Object);

 protected:
  void AppendTop(const caffe::NetParameter& param, int layer_id,
                 int top_id, std::set<std::string>* available_blobs,
                 std::map<std::string, int>* blob_name_to_idx);
  int AppendBottom(const caffe::NetParameter& param, int layer_id,
                   int bottom_id, std::set<std::string>* available_blobs,
                   std::map<std::string, int>* blob_name_to_idx);

  std::string name_;
  std::vector<ObjectPtr<Layer>> layers_;
  std::vector<std::string> layer_names_;
  std::map<std::string, int> layer_names_index_;
  std::vector<ObjectPtr<Blob>> blobs_;
  std::vector<std::string> blob_names_;
  std::map<std::string, int> blob_names_index_;
  std::vector<std::vector<Blob*>> bottom_vecs_;
  std::vector<std::vector<Blob*>> top_vecs_;
  std::vector<int> net_input_blob_indices_;
  std::vector<int> net_output_blob_indices_;
  std::vector<Blob*> net_input_blobs_;
  std::vector<Blob*> net_output_blobs_;
  std::vector<std::string> net_input_blob_names_;
  std::vector<std::string> net_output_blob_names_;

 private:
  Net(const Net&) = delete;
  Net& operator=(const Net&) = delete;
};

/**
 * @brief Read network parameters from a text-format .prototxt file.
 * @param filename Path to the .prototxt file.
 * @return Parsed NetParameter protobuf message.
 * @throws ValueError if the file cannot be read or parsed.
 */
caffe::NetParameter ReadNetParamsFromTextFile(const std::string& filename);
/**
 * @brief Parse network parameters from a text-format prototxt string.
 * @param text Text-format protobuf string.
 * @return Parsed NetParameter protobuf message.
 * @throws ValueError if the string cannot be parsed.
 * @note Defined in the DLL to avoid cross-DLL protobuf static initialization issues.
 */
caffe::NetParameter ReadNetParamsFromTextString(const std::string& text);
/**
 * @brief Read network parameters from a binary .caffemodel file.
 * @param filename Path to the binary .caffemodel file.
 * @return Parsed NetParameter protobuf message.
 * @throws ValueError if the file cannot be read or parsed.
 */
caffe::NetParameter ReadNetParamsFromBinaryFile(const std::string& filename);

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_NET_HPP_
