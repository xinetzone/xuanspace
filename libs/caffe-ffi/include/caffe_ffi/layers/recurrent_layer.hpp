#ifndef CAFFE_FFI_LAYERS_RECURRENT_LAYER_HPP_
#define CAFFE_FFI_LAYERS_RECURRENT_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Abstract base class for recurrent layers (vanilla RNN, LSTM).
 *
 * RecurrentLayer factors out the time-step unrolling shared by all recurrent
 * cells. It owns:
 *   - the hidden state h_{t-1} (and, for LSTM, the cell state c_{t-1}) carried
 *     across the unrolled time steps,
 *   - the recurrent gradient buffer dh_next (and dc_next) used by BPTT,
 *   - the per-step iteration loop in Forward_cpu / Backward_cpu.
 *
 * Subclasses only implement the single-step cell math via the pure-virtual
 * hooks:
 *   - LayerSetUpStep()  : create/validate the weight blobs and set H_.
 *   - ReshapeStep()     : allocate per-step caches (depends on T_, N_, D_, H_).
 *   - ForwardStep(t, x_t, h_t) : compute h_t for step t from x_t and h_prev_.
 *   - BackwardStep(t, x_t, h_prev, dy_t, dx_t, dh_next) : BPTT for step t.
 *
 * Input/output layout follows the Caffe convention: the single bottom blob is
 * a 3-D time-first sequence bottom[0] = (T, N, D) (T timesteps × N batch × D
 * input dim), and the single top blob collects every time-step hidden state
 * top[0] = (T, N, H). T is taken from the actual input length bottom[0]->shape(0);
 * num_steps (from recurrent_param) is read for logging/compatibility.
 */
class RecurrentLayer : public Layer {
 public:
  static constexpr int _type_child_slots = 2;

  explicit RecurrentLayer(const caffe::LayerParameter& param) : Layer(param) {
    h_prev_ = make_object<Blob>();
    c_prev_ = make_object<Blob>();
    h_t_ = make_object<Blob>();
    dh_next_ = make_object<Blob>();
    dc_next_ = make_object<Blob>();
  }

  void LayerSetUp(const std::vector<Blob*>& bottom,
                  const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom,
               const std::vector<Blob*>& top) override;
  void Forward_cpu(const std::vector<Blob*>& bottom,
                   const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }
  const char* type() const override { return "Recurrent"; }

  TVM_FFI_DECLARE_OBJECT_INFO("caffe_ffi.RecurrentLayer", RecurrentLayer, Layer);

 protected:
  // ---- Pure-virtual single-step hooks (implemented by subclasses) ----
  // Create/validate the weight blobs and set H_. Called once in LayerSetUp.
  virtual void LayerSetUpStep(const std::vector<Blob*>& bottom) = 0;
  // Allocate per-step caches. Called from Reshape after T_/N_/D_/H_ are known.
  virtual void ReshapeStep() = 0;
  // Forward one step: compute h_t from x_t and the current h_prev_/c_prev_.
  virtual void ForwardStep(int t, const float* x_t, float* h_t) = 0;
  // Backward one step (BPTT). Reads/updates dh_next (and dc_next via members).
  virtual void BackwardStep(int t, const float* x_t, const float* h_prev,
                            const float* dy_t, float* dx_t, float* dh_next) = 0;
  // Optional hook to reset internal gradient accumulators before the BPTT loop.
  virtual void BackwardStart() {}

  // ---- Dimensions ----
  int num_steps_ = 1;  // configured unroll length (recurrent_param().num_steps())
  int T_ = 0;          // actual number of timesteps (input length)
  int N_ = 0;          // batch size
  int D_ = 0;          // input feature dimension
  int H_ = 0;          // hidden dimension (set by subclass)
  bool expose_hidden_ = false;

  // ---- State buffers ----
  ObjectPtr<Blob> h_prev_;   // (N, H) hidden state carried across steps
  ObjectPtr<Blob> c_prev_;   // (N, H) cell state carried across steps (LSTM)
  ObjectPtr<Blob> h_t_;      // (N, H) working buffer / zero buffer for t==0
  ObjectPtr<Blob> dh_next_;  // (N, H) recurrent gradient for BPTT
  ObjectPtr<Blob> dc_next_;  // (N, H) recurrent cell gradient for BPTT (LSTM)
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_RECURRENT_LAYER_HPP_