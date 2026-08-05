#ifndef CAFFE_FFI_LAYERS_DATA_IO_BRIDGE_HPP_
#define CAFFE_FFI_LAYERS_DATA_IO_BRIDGE_HPP_

#include <string>
#include <vector>

#include "caffe_ffi/blob.hpp"

namespace caffe_ffi {

/**
 * @brief Shared Python/numpy bridge for the data I/O layers.
 *
 * The data I/O layers (Data, ImageData, HDF5Data, HDF5Output, WindowData) read
 * or write their payload through a Python-side callback registered via the FFI
 * function `caffe_ffi.data_io.register`. The callback is a TVM FFI Function
 * stored in a static registry keyed by "<layer_type>.<layer_name>".
 *
 * On Forward the callback is invoked with the (mutable) data tensors of the
 * relevant blobs (Array<Tensor>, DLPack interop) so the Python side can fill
 * the outputs (data-source layers) or read the inputs (output layers).
 *
 * If no callback is registered for a key, InvokeDataIOCallback returns false
 * and the caller should degrade to a no-op (output zeros).
 */

/**
 * @brief Register a data-source/output callback under the given key.
 * \note Called from the Python side through the FFI function
 *       `caffe_ffi.data_io.register`. Defined in data_io_bridge.cpp.
 */
void RegisterDataIOCallback(const std::string& key, Function callback);

/**
 * @brief Look up a registered data-source/output callback by key.
 * \return The callback, or an undefined Function if not registered.
 */
Function LookupDataIOCallback(const std::string& key);

/**
 * @brief Invoke the callback (if any) for `key` with the given blobs' tensors.
 * \param key      Registration key ("<layer_type>.<layer_name>").
 * \param blobs    Blobs whose data tensors are passed to the callback.
 * \param writable When true use mutable_data_tensor() (COW write view) for the
 *                 Python side to fill; when false use data_tensor() (read view).
 * \return true if a callback was found and invoked, false otherwise.
 */
bool InvokeDataIOCallback(const std::string& key, const std::vector<Blob*>& blobs,
                          bool writable);

/**
 * @brief Build the registration key for a layer: "<type>.<name>".
 */
inline std::string DataIOKey(const char* type, const std::string& name) {
  return std::string(type) + "." + name;
}

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_DATA_IO_BRIDGE_HPP_