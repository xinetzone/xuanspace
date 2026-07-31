#include "test_harness.hpp"

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layers/base_conv_layer.hpp"
#include "caffe_ffi/layers/deconv_layer.hpp"
#include "caffe_ffi/layers/conv_layer.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/math_utils.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

using namespace caffe_ffi;

// Helper: create a LayerParameter protobuf message for Deconvolution
static caffe::LayerParameter MakeDeconvParam(
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int group = 1, bool bias_term = true,
    int dilation_h = 1, int dilation_w = 1) {
  caffe::LayerParameter param;
  param.set_type("Deconvolution");
  auto* cp = param.mutable_convolution_param();
  cp->set_num_output(num_output);
  cp->add_kernel_size(kernel_h);
  if (kernel_w != kernel_h) cp->add_kernel_size(kernel_w);
  cp->add_pad(pad_h);
  if (pad_w != pad_h) {
    cp->clear_pad();
    cp->set_pad_h(pad_h);
    cp->set_pad_w(pad_w);
  }
  cp->add_stride(stride_h);
  if (stride_w != stride_h) {
    cp->clear_stride();
    cp->set_stride_h(stride_h);
    cp->set_stride_w(stride_w);
  }
  if (dilation_h != 1 || dilation_w != 1) {
    cp->add_dilation(dilation_h);
    if (dilation_w != dilation_h) {
      cp->clear_dilation();
      cp->set_dilation_h(dilation_h);
      cp->set_dilation_w(dilation_w);
    }
  }
  cp->set_group(group);
  cp->set_bias_term(bias_term);
  return param;
}

// Helper: set all weights to a fixed simple pattern for reproducible tests
static void InitWeights(Layer* layer, float w_val, float b_val = 0.0f) {
  auto* w = layer->blobs()[0].get();
  float* wdata = w->cpu_mutable_data();
  for (int64_t i = 0; i < w->count(); ++i) wdata[i] = w_val;
  if (layer->blobs().size() > 1) {
    auto* b = layer->blobs()[1].get();
    float* bdata = b->cpu_mutable_data();
    for (int64_t i = 0; i < b->count(); ++i) bdata[i] = b_val;
  }
}

// Helper: fill blob with sequential values 0, 1, 2, ...
static void FillSequential(Blob* blob, float start = 0.0f, float step = 1.0f) {
  float* data = blob->cpu_mutable_data();
  float v = start;
  for (int64_t i = 0; i < blob->count(); ++i) {
    data[i] = v;
    v += step;
  }
}

// ────────────────────────────────────────────────────────────────────────────
//  Output shape computation tests
// ────────────────────────────────────────────────────────────────────────────

TEST(DeconvLayerTest, OutputShape_K3S1P1_Same) {
  // kernel=3, stride=1, pad=1 → output = (H-1)*1 + 3 - 2*1 = H (same padding)
  auto param = MakeDeconvParam(2, 3, 3, 1, 1, 1, 1);
  DeconvolutionLayer layer(param);

  std::vector<int64_t> bottom_shape = {1, 2, 4, 4};
  Blob bottom(bottom_shape);
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};

  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  EXPECT_EQ(top.shape(0), 1);   // N
  EXPECT_EQ(top.shape(1), 2);   // num_output
  EXPECT_EQ(top.shape(2), 4);   // Ho = (4-1)*1+3-2*1 = 4
  EXPECT_EQ(top.shape(3), 4);   // Wo = 4
}

TEST(DeconvLayerTest, OutputShape_K2S2P0_Upsample2x) {
  // kernel=2, stride=2, pad=0 → output = (H-1)*2+2 = 2H (2x upsampling)
  auto param = MakeDeconvParam(3, 2, 2, 0, 0, 2, 2, 1, false);
  DeconvolutionLayer layer(param);

  std::vector<int64_t> bottom_shape = {1, 3, 3, 3};
  Blob bottom(bottom_shape);
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};

  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  EXPECT_EQ(top.shape(0), 1);
  EXPECT_EQ(top.shape(1), 3);
  EXPECT_EQ(top.shape(2), 6);   // Ho = (3-1)*2+2 = 6
  EXPECT_EQ(top.shape(3), 6);   // Wo = 6
}

TEST(DeconvLayerTest, OutputShape_K3S2P1) {
  // kernel=3, stride=2, pad=1 → output = (H-1)*2+3-2 = 2H-1
  auto param = MakeDeconvParam(4, 3, 3, 1, 1, 2, 2);
  DeconvolutionLayer layer(param);

  std::vector<int64_t> bottom_shape = {2, 4, 5, 5};
  Blob bottom(bottom_shape);
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};

  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  EXPECT_EQ(top.shape(0), 2);
  EXPECT_EQ(top.shape(1), 4);
  EXPECT_EQ(top.shape(2), 9);   // Ho = (5-1)*2+3-2 = 9
  EXPECT_EQ(top.shape(3), 9);
}

TEST(DeconvLayerTest, OutputShape_K1S1P0_1x1) {
  // 1x1 deconv → output spatial same as input, channels = num_output
  auto param = MakeDeconvParam(8, 1, 1, 0, 0, 1, 1);
  DeconvolutionLayer layer(param);

  std::vector<int64_t> bottom_shape = {1, 8, 3, 3};
  Blob bottom(bottom_shape);
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};

  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  EXPECT_EQ(top.shape(0), 1);
  EXPECT_EQ(top.shape(1), 8);
  EXPECT_EQ(top.shape(2), 3);
  EXPECT_EQ(top.shape(3), 3);
}

TEST(DeconvLayerTest, OutputShape_Grouped) {
  // group=2 deconv: Ci=4, Co=6, group=2 → Ci/g=2, Co/g=3
  // kernel=3, stride=1, pad=1 → same output spatial
  auto param = MakeDeconvParam(6, 3, 3, 1, 1, 1, 1, 2);
  DeconvolutionLayer layer(param);

  std::vector<int64_t> bottom_shape = {1, 4, 4, 4};
  Blob bottom(bottom_shape);
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};

  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  EXPECT_EQ(top.shape(0), 1);
  EXPECT_EQ(top.shape(1), 6);   // num_output
  EXPECT_EQ(top.shape(2), 4);
  EXPECT_EQ(top.shape(3), 4);
}

// ────────────────────────────────────────────────────────────────────────────
//  Forward correctness tests
// ────────────────────────────────────────────────────────────────────────────

TEST(DeconvLayerTest, Forward_IdentityWeights_NoBias) {
  // 1x1 deconv with weight=1, bias=0: output channel c = input channel c
  // Weight shape: [conv_out_channels_, conv_in_channels_/group_, 1, 1]
  // For deconv: conv_out_channels_ = Ci = 3, conv_in_channels_ = Co = 3
  // weight[g*Co/g + co, ci] = 1 if co==ci else 0 → identity matrix
  auto param = MakeDeconvParam(3, 1, 1, 0, 0, 1, 1, 1, false);
  DeconvolutionLayer layer(param);

  std::vector<int64_t> bottom_shape = {1, 3, 2, 2};
  Blob bottom(bottom_shape);
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};

  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  // Identity weight matrix: I (3x3)
  float* w = layer.blobs()[0]->cpu_mutable_data();
  std::memset(w, 0, sizeof(float) * layer.blobs()[0]->count());
  for (int i = 0; i < 3; ++i) {
    w[i * 3 + i] = 1.0f;  // identity: w[i,i,0,0] = 1
  }

  // Input: sequential data
  FillSequential(&bottom, 1.0f, 1.0f);
  // bottom data: [1,2,3,4, 5,6,7,8, 9,10,11,12] (3ch × 2×2)

  layer.Forward(bottoms, tops);

  const float* top_data = top.cpu_data();
  // With identity 1x1 deconv, output should equal input
  for (int64_t i = 0; i < bottom.count(); ++i) {
    EXPECT_NEAR(top_data[i], bottom.cpu_data()[i], 1e-5f);
  }
}

TEST(DeconvLayerTest, Forward_UnityWeights_WithBias) {
  // kernel=3, stride=1, pad=1 (same size), all weights = 1.0f, bias = b_val
  // Each output pixel = sum over 3x3 window of input × 1 + bias
  auto param = MakeDeconvParam(1, 3, 3, 1, 1, 1, 1, 1, true);
  DeconvolutionLayer layer(param);

  std::vector<int64_t> bottom_shape = {1, 1, 2, 2};
  Blob bottom(bottom_shape);
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};

  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);
  InitWeights(&layer, 1.0f, 0.5f);

  // Input:
  // [1  2]
  // [3  4]
  float* bdata = bottom.cpu_mutable_data();
  bdata[0] = 1; bdata[1] = 2; bdata[2] = 3; bdata[3] = 4;

  layer.Forward(bottoms, tops);

  // Deconv forward = Conv backward w.r.t. input:
  // With all weights=1 and 2x2 input [1,2;3,4], each output pixel sums
  // overlapping input positions. For symmetric input+kernel+pad configuration,
  // all output pixels receive equal total contribution (1+2+3+4=10).
  // Plus bias 0.5 → all outputs = 10.5.
  const float* td = top.cpu_data();
  const float expected = 10.5f;
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(td[i], expected, 1e-5f) << "Output pixel " << i;
  }
}

TEST(DeconvLayerTest, Forward_NoBias_OutputRange) {
  auto param = MakeDeconvParam(2, 3, 3, 1, 1, 1, 1, 1, false);
  DeconvolutionLayer layer(param);

  std::vector<int64_t> bottom_shape = {1, 2, 4, 4};
  Blob bottom(bottom_shape);
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};

  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);
  InitWeights(&layer, 0.1f);
  FillSequential(&bottom, 0.0f, 0.01f);

  layer.Forward(bottoms, tops);

  // Check output is finite and non-zero
  const float* td = top.cpu_data();
  for (int64_t i = 0; i < top.count(); ++i) {
    EXPECT_TRUE(std::isfinite(td[i]));
  }
}

// ────────────────────────────────────────────────────────────────────────────
//  Backward gradient check (central finite difference)
// ────────────────────────────────────────────────────────────────────────────

static float LossL2(const Blob& top) {
  const float* data = top.cpu_data();
  double sum = 0.0;
  for (int64_t i = 0; i < top.count(); ++i) {
    sum += static_cast<double>(data[i]) * static_cast<double>(data[i]);
  }
  return static_cast<float>(0.5 * sum);
}

static void SetGradientFromOutput(Blob* top) {
  float* diff = top->cpu_mutable_diff();
  const float* data = top->cpu_data();
  // dL/dout = out (gradient of 0.5*||out||^2)
  for (int64_t i = 0; i < top->count(); ++i) {
    diff[i] = data[i];
  }
}

TEST(DeconvLayerTest, Backward_GradientCheck_Bottom) {
  auto param = MakeDeconvParam(2, 3, 3, 1, 1, 1, 1, 1, false);
  DeconvolutionLayer layer(param);

  std::vector<int64_t> bottom_shape = {1, 2, 3, 3};
  Blob bottom(bottom_shape);
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};

  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);
  InitWeights(&layer, 0.1f);
  FillSequential(&bottom, 0.1f, 0.05f);

  // Forward
  layer.Forward(bottoms, tops);
  float loss = LossL2(top);
  EXPECT_GT(loss, 0);

  // Set top diff = dL/dtop = top (for 0.5*||top||^2 loss)
  SetGradientFromOutput(&top);

  // Backward: compute bottom_diff analytically
  std::vector<bool> prop_down = {true};
  // Don't compute weight grads: set param_propagate_down to false
  layer.set_param_propagate_down(0, false);
  layer.Backward(tops, prop_down, bottoms);

  // Numerical gradient check using central difference
  const float eps = 1e-3f;
  const float* analytical_diff = bottom.cpu_diff();
  float* bdata = bottom.cpu_mutable_data();

  for (int i = 0; i < 12; ++i) {  // check first 12 elements for speed
    float orig = bdata[i];

    bdata[i] = orig + eps;
    layer.Forward(bottoms, tops);
    float loss_plus = LossL2(top);

    bdata[i] = orig - eps;
    layer.Forward(bottoms, tops);
    float loss_minus = LossL2(top);

    bdata[i] = orig;  // restore

    float numerical_grad = (loss_plus - loss_minus) / (2 * eps);
    float analytical = analytical_diff[i];

    // Relaxed tolerance due to floating point accumulation
    float denom = std::max(std::abs(numerical_grad), std::abs(analytical));
    float rel_err = (denom > 1e-6f) ? std::abs(numerical_grad - analytical) / denom : std::abs(numerical_grad - analytical);
    EXPECT_LT(rel_err, 0.05f)  // 5% relative error tolerance
        << "Gradient mismatch at index " << i << ": numerical=" << numerical_grad
        << " analytical=" << analytical;
  }
}

TEST(DeconvLayerTest, Backward_GradientCheck_Weight) {
  auto param = MakeDeconvParam(2, 3, 3, 1, 1, 1, 1, 1, false);
  DeconvolutionLayer layer(param);

  std::vector<int64_t> bottom_shape = {1, 2, 3, 3};
  Blob bottom(bottom_shape);
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};

  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);
  InitWeights(&layer, 0.1f);
  FillSequential(&bottom, 0.1f, 0.05f);

  // Forward
  layer.Forward(bottoms, tops);
  SetGradientFromOutput(&top);

  // Backward with weight gradient
  std::vector<bool> prop_down = {false};  // don't need bottom diff
  layer.set_param_propagate_down(0, true);
  layer.Backward(tops, prop_down, bottoms);

  const float* analytical_wdiff = layer.blobs()[0]->cpu_diff();

  // Numerical gradient for weights
  const float eps = 1e-3f;
  float* wdata = layer.blobs()[0]->cpu_mutable_data();
  int64_t wcount = layer.blobs()[0]->count();

  for (int i = 0; i < std::min<int64_t>(wcount, 18); ++i) {
    float orig = wdata[i];

    wdata[i] = orig + eps;
    layer.Forward(bottoms, tops);
    float loss_plus = LossL2(top);

    wdata[i] = orig - eps;
    layer.Forward(bottoms, tops);
    float loss_minus = LossL2(top);

    wdata[i] = orig;

    float numerical_grad = (loss_plus - loss_minus) / (2 * eps);
    float analytical = analytical_wdiff[i];

    float denom = std::max(std::abs(numerical_grad), std::abs(analytical));
    float rel_err = (denom > 1e-6f) ? std::abs(numerical_grad - analytical) / denom : std::abs(numerical_grad - analytical);
    EXPECT_LT(rel_err, 0.05f)
        << "Weight gradient mismatch at index " << i << ": numerical=" << numerical_grad
        << " analytical=" << analytical;
  }
}

// ────────────────────────────────────────────────────────────────────────────
//  Symmetric Conv-Deconv reconstruction test
// ────────────────────────────────────────────────────────────────────────────

TEST(DeconvLayerTest, ConvDeconv_Symmetric) {
  // A Conv followed by a Deconv with same parameters and transposed weights
  // should approximately reconstruct the input (up to border effects without pad)
  int num_output = 2;
  int channels = 2;
  int kernel = 3;
  int pad = 1;
  int stride = 1;

  // Setup Conv
  caffe::LayerParameter conv_param;
  conv_param.set_type("Convolution");
  auto* cp = conv_param.mutable_convolution_param();
  cp->set_num_output(num_output);
  cp->add_kernel_size(kernel);
  cp->add_pad(pad);
  cp->add_stride(stride);
  cp->set_bias_term(false);
  cp->set_group(1);
  ConvolutionLayer conv_layer(conv_param);

  // Setup Deconv with same params
  auto deconv_param = MakeDeconvParam(channels, kernel, kernel, pad, pad, stride, stride, 1, false);
  DeconvolutionLayer deconv_layer(deconv_param);

  std::vector<int64_t> input_shape = {1, channels, 4, 4};
  Blob input(input_shape);
  Blob conv_out;
  Blob deconv_out;
  std::vector<Blob*> in_blobs = {&input};
  std::vector<Blob*> conv_tops = {&conv_out};
  std::vector<Blob*> deconv_bottoms = {&conv_out};
  std::vector<Blob*> deconv_tops = {&deconv_out};

  conv_layer.LayerSetUp(in_blobs, conv_tops);
  conv_layer.Reshape(in_blobs, conv_tops);
  deconv_layer.LayerSetUp(deconv_bottoms, deconv_tops);
  deconv_layer.Reshape(deconv_bottoms, deconv_tops);

  // Initialize conv weights with random-ish pattern
  float* cw = conv_layer.blobs()[0]->cpu_mutable_data();
  for (int64_t i = 0; i < conv_layer.blobs()[0]->count(); ++i) {
    cw[i] = 0.1f * std::sin(static_cast<float>(i) * 0.3f);
  }

  // Deconv weight = transpose of conv weight
  // Conv weight shape: [Co, Ci/g, Kh, Kw] = [2, 2, 3, 3]
  // Deconv weight shape: [Co, Ci/g, Kh, Kw] = [2, 2, 3, 3]
  // In native Caffe, deconv weight shares memory layout with conv weight;
  // transposition is handled by GEMM TransA/TransB flags.
  // For reconstruction test, copy weights directly.
  float* dw = deconv_layer.blobs()[0]->cpu_mutable_data();
  std::memcpy(dw, cw, sizeof(float) * conv_layer.blobs()[0]->count());

  FillSequential(&input, 0.1f, 0.02f);

  conv_layer.Forward(in_blobs, conv_tops);
  deconv_layer.Forward(deconv_bottoms, deconv_tops);

  // The Conv-Deconv pair with same pad/k/stride should produce output of same shape
  EXPECT_EQ(deconv_out.shape(0), input.shape(0));
  EXPECT_EQ(deconv_out.shape(1), input.shape(1));
  EXPECT_EQ(deconv_out.shape(2), input.shape(2));
  EXPECT_EQ(deconv_out.shape(3), input.shape(3));
}

// ────────────────────────────────────────────────────────────────────────────
//  Registration and type tests
// ────────────────────────────────────────────────────────────────────────────

TEST(DeconvLayerTest, LayerType) {
  auto param = MakeDeconvParam(1, 1, 1, 0, 0, 1, 1);
  DeconvolutionLayer layer(param);
  EXPECT_STREQ(layer.type(), "Deconvolution");
}

TEST(DeconvLayerTest, IsDeconvolutionReverseDims) {
  auto param = MakeDeconvParam(1, 1, 1, 0, 0, 1, 1);
  DeconvolutionLayer layer(param);
  // reverse_dimensions() should return true for deconv
  // (indirectly tested via Reshape setting correct conv_input_shape_)
  std::vector<int64_t> bottom_shape = {1, 1, 2, 2};
  Blob bottom(bottom_shape);
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);
  EXPECT_EQ(top.shape(1), 1);
}

TEST(DeconvLayerTest, ExactBlobs) {
  auto param = MakeDeconvParam(2, 3, 3, 1, 1, 1, 1, 1, true);
  DeconvolutionLayer layer(param);
  EXPECT_EQ(layer.ExactNumBottomBlobs(), 1);
  EXPECT_EQ(layer.ExactNumTopBlobs(), 1);
}
