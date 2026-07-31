#include "test_harness.hpp"

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layers/neuron_layer.hpp"
#include "caffe_ffi/layers/relu_layer.hpp"
#include "caffe_ffi/layers/sigmoid_layer.hpp"
#include "caffe_ffi/layers/tanh_layer.hpp"
#include "caffe_ffi/layers/elu_layer.hpp"
#include "caffe_ffi/layers/prelu_layer.hpp"
#include "caffe_ffi/fill.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

using namespace caffe_ffi;

// Helper: create a minimal LayerParameter with just the type set
static caffe::LayerParameter MakeLayerParam(const std::string& type) {
  caffe::LayerParameter param;
  param.set_type(type);
  return param;
}

// Helper: create a ReLU LayerParameter with negative_slope
static caffe::LayerParameter MakeReLUParam(float negative_slope = 0.0f) {
  auto param = MakeLayerParam("ReLU");
  param.mutable_relu_param()->set_negative_slope(negative_slope);
  return param;
}

// Helper: create an ELU LayerParameter with alpha
static caffe::LayerParameter MakeELUParam(float alpha = 1.0f) {
  auto param = MakeLayerParam("ELU");
  param.mutable_elu_param()->set_alpha(alpha);
  return param;
}

// Helper: create a PReLU LayerParameter
static caffe::LayerParameter MakePReLUParam(bool channel_shared = true, float slope = 0.25f) {
  auto param = MakeLayerParam("PReLU");
  auto* p = param.mutable_prelu_param();
  p->set_channel_shared(channel_shared);
  return param;
}

// Helper: set blob data to specific values
static void SetBlobData(Blob* blob, const std::vector<float>& values) {
  ASSERT_EQ(blob->count(), static_cast<int64_t>(values.size()));
  float* data = blob->cpu_mutable_data();
  std::memcpy(data, values.data(), sizeof(float) * values.size());
}

// Helper: set top diff to all-ones for gradient checking
static void SetTopDiffToOnes(Blob* top) {
  float* diff = top->cpu_mutable_diff();
  for (int64_t i = 0; i < top->count(); ++i) diff[i] = 1.0f;
}

// Helper: numerical gradient check using central difference
static void CheckGradient(const std::function<float()>& loss_fn,
                          Blob* bottom,
                          const float* analytical_diff,
                          int n_check = 8,
                          float eps = 1e-3f,
                          float tol = 0.05f) {
  float* bdata = bottom->cpu_mutable_data();
  for (int i = 0; i < std::min<int>(n_check, static_cast<int>(bottom->count())); ++i) {
    float orig = bdata[i];

    bdata[i] = orig + eps;
    float loss_plus = loss_fn();

    bdata[i] = orig - eps;
    float loss_minus = loss_fn();

    bdata[i] = orig;

    float numerical_grad = (loss_plus - loss_minus) / (2 * eps);
    float analytical = analytical_diff[i];

    float denom = std::max(std::abs(numerical_grad), std::abs(analytical));
    float rel_err = (denom > 1e-6f) ? std::abs(numerical_grad - analytical) / denom : std::abs(numerical_grad - analytical);
    EXPECT_LT(rel_err, tol)
        << "Gradient mismatch at index " << i << ": numerical=" << numerical_grad
        << " analytical=" << analytical << " (orig_val=" << orig << ")";
  }
}

// ────────────────────────────────────────────────────────────────────────────
//  NeuronLayer base class tests
// ────────────────────────────────────────────────────────────────────────────

// We can't instantiate NeuronLayer directly (it's abstract), so we test
// through a concrete subclass (ReLULayer) which inherits NeuronLayer behavior.

TEST(NeuronLayerTest, ExactBlobs) {
  ReLULayer layer(MakeReLUParam());
  EXPECT_EQ(layer.ExactNumBottomBlobs(), 1);
  EXPECT_EQ(layer.ExactNumTopBlobs(), 1);
}

TEST(NeuronLayerTest, ReshapeMatchesBottomShape) {
  ReLULayer layer(MakeReLUParam());
  Blob bottom(std::vector<int64_t>{2, 3, 4, 5});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);
  EXPECT_EQ(top.num_axes(), 4);
  EXPECT_EQ(top.shape(0), 2);
  EXPECT_EQ(top.shape(1), 3);
  EXPECT_EQ(top.shape(2), 4);
  EXPECT_EQ(top.shape(3), 5);
  EXPECT_EQ(top.count(), bottom.count());
}

TEST(NeuronLayerTest, Reshape1D) {
  ReLULayer layer(MakeReLUParam());
  Blob bottom(std::vector<int64_t>{10});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);
  EXPECT_EQ(top.num_axes(), 1);
  EXPECT_EQ(top.shape(0), 10);
  EXPECT_EQ(top.count(), 10);
}

TEST(NeuronLayerTest, Reshape2D) {
  ReLULayer layer(MakeReLUParam());
  Blob bottom(std::vector<int64_t>{3, 7});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);
  EXPECT_EQ(top.num_axes(), 2);
  EXPECT_EQ(top.shape(0), 3);
  EXPECT_EQ(top.shape(1), 7);
}

// ────────────────────────────────────────────────────────────────────────────
//  ReLU tests
// ────────────────────────────────────────────────────────────────────────────

TEST(ReLULayerTest, TypeName) {
  ReLULayer layer(MakeReLUParam());
  EXPECT_STREQ(layer.type(), "ReLU");
}

TEST(ReLULayerTest, ForwardStandard) {
  // Standard ReLU: negative_slope=0, y = max(x, 0)
  ReLULayer layer(MakeReLUParam(0.0f));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 6});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-2.0f, -1.0f, -0.5f, 0.0f, 1.0f, 3.0f});
  layer.Forward(bottoms, tops);

  const float* y = top.cpu_data();
  EXPECT_FLOAT_EQ(y[0], 0.0f);   // -2 → 0
  EXPECT_FLOAT_EQ(y[1], 0.0f);   // -1 → 0
  EXPECT_FLOAT_EQ(y[2], 0.0f);   // -0.5 → 0
  EXPECT_FLOAT_EQ(y[3], 0.0f);   // 0 → 0
  EXPECT_FLOAT_EQ(y[4], 1.0f);   // 1 → 1
  EXPECT_FLOAT_EQ(y[5], 3.0f);   // 3 → 3
}

TEST(ReLULayerTest, ForwardLeakyReLU) {
  // Leaky ReLU: negative_slope=0.1, y = max(x,0) + 0.1*min(x,0)
  ReLULayer layer(MakeReLUParam(0.1f));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 4});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-2.0f, -0.5f, 0.0f, 5.0f});
  layer.Forward(bottoms, tops);

  const float* y = top.cpu_data();
  EXPECT_NEAR(y[0], -0.2f, 1e-6f);  // -2 * 0.1 = -0.2
  EXPECT_NEAR(y[1], -0.05f, 1e-6f); // -0.5 * 0.1 = -0.05
  EXPECT_NEAR(y[2], 0.0f, 1e-6f);
  EXPECT_NEAR(y[3], 5.0f, 1e-6f);
}

TEST(ReLULayerTest, BackwardStandard) {
  ReLULayer layer(MakeReLUParam(0.0f));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 4});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-1.0f, 0.0f, 0.5f, 2.0f});
  layer.Forward(bottoms, tops);
  SetTopDiffToOnes(&top);

  std::vector<bool> prop_down = {true};
  layer.Backward(tops, prop_down, bottoms);

  const float* dx = bottom.cpu_diff();
  EXPECT_FLOAT_EQ(dx[0], 0.0f);  // x<0, slope=0
  EXPECT_FLOAT_EQ(dx[1], 0.0f);  // x=0, slope=0 (negative_slope=0)
  EXPECT_FLOAT_EQ(dx[2], 1.0f);  // x>0, slope=1
  EXPECT_FLOAT_EQ(dx[3], 1.0f);  // x>0, slope=1
}

TEST(ReLULayerTest, BackwardLeaky) {
  ReLULayer layer(MakeReLUParam(0.2f));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 3});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-3.0f, 0.0f, 4.0f});
  layer.Forward(bottoms, tops);
  SetTopDiffToOnes(&top);

  std::vector<bool> prop_down = {true};
  layer.Backward(tops, prop_down, bottoms);

  const float* dx = bottom.cpu_diff();
  EXPECT_FLOAT_EQ(dx[0], 0.2f);  // x<0, slope=0.2
  EXPECT_FLOAT_EQ(dx[1], 0.2f);  // x=0, slope=negative_slope
  EXPECT_FLOAT_EQ(dx[2], 1.0f);  // x>0, slope=1
}

TEST(ReLULayerTest, BackwardSkipWhenPropagateDownFalse) {
  ReLULayer layer(MakeReLUParam());
  Blob bottom(std::vector<int64_t>{1, 1, 1, 3});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {1.0f, 2.0f, 3.0f});
  layer.Forward(bottoms, tops);

  // Fill bottom_diff with sentinel value to detect if Backward writes to it
  float* bdiff = bottom.cpu_mutable_diff();
  for (int64_t i = 0; i < bottom.count(); ++i) bdiff[i] = 999.0f;

  std::vector<bool> prop_down = {false};
  layer.Backward(tops, prop_down, bottoms);

  // bottom_diff should be unchanged
  for (int64_t i = 0; i < bottom.count(); ++i) {
    EXPECT_FLOAT_EQ(bdiff[i], 999.0f);
  }
}

TEST(ReLULayerTest, GradientCheck) {
  ReLULayer layer(MakeReLUParam(0.1f));
  Blob bottom(std::vector<int64_t>{1, 1, 2, 2});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  // Mixed positive/negative values for gradient check (avoid near-zero outputs
  // to keep float32 finite-difference relative error stable)
  float* bdata = bottom.cpu_mutable_data();
  float vals[] = {-1.5f, -1.0f, 0.5f, 2.0f};
  std::memcpy(bdata, vals, sizeof(vals));

  auto forward_loss = [&]() {
    layer.Forward(bottoms, tops);
    float loss = 0.0f;
    const float* td = top.cpu_data();
    for (int64_t i = 0; i < top.count(); ++i) loss += td[i] * td[i] * 0.5f;
    // Set top_diff = dL/dy = y
    float* tdiff = top.cpu_mutable_diff();
    for (int64_t i = 0; i < top.count(); ++i) tdiff[i] = td[i];
    return loss;
  };

  float loss = forward_loss();
  EXPECT_GT(loss, 0.0f);

  std::vector<bool> prop_down = {true};
  layer.Backward(tops, prop_down, bottoms);

  // Higher tolerance (7%) for ReLU with negative_slope due to kink at 0 causing
  // larger finite-difference error near the transition point for small gradients
  CheckGradient([&]() { return forward_loss(); }, &bottom, bottom.cpu_diff(), 4, 1e-3f, 0.07f);
}

// ────────────────────────────────────────────────────────────────────────────
//  Sigmoid tests
// ────────────────────────────────────────────────────────────────────────────

TEST(SigmoidLayerTest, TypeName) {
  SigmoidLayer layer(MakeLayerParam("Sigmoid"));
  EXPECT_STREQ(layer.type(), "Sigmoid");
}

TEST(SigmoidLayerTest, ForwardKnownValues) {
  SigmoidLayer layer(MakeLayerParam("Sigmoid"));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 5});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f});
  layer.Forward(bottoms, tops);

  const float* y = top.cpu_data();
  EXPECT_NEAR(y[0], 0.0f, 1e-4f);       // sigmoid(-10) ≈ 0
  EXPECT_NEAR(y[1], 0.2689f, 1e-3f);    // sigmoid(-1) ≈ 0.2689
  EXPECT_NEAR(y[2], 0.5f, 1e-6f);       // sigmoid(0) = 0.5
  EXPECT_NEAR(y[3], 0.7311f, 1e-3f);    // sigmoid(1) ≈ 0.7311
  EXPECT_NEAR(y[4], 1.0f, 1e-4f);       // sigmoid(10) ≈ 1
}

TEST(SigmoidLayerTest, ForwardRange) {
  SigmoidLayer layer(MakeLayerParam("Sigmoid"));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 100});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  float* bdata = bottom.cpu_mutable_data();
  for (int i = 0; i < 100; ++i) bdata[i] = -5.0f + 0.1f * i;
  layer.Forward(bottoms, tops);

  const float* y = top.cpu_data();
  for (int i = 0; i < 100; ++i) {
    EXPECT_GT(y[i], 0.0f);
    EXPECT_LT(y[i], 1.0f);
  }
}

TEST(SigmoidLayerTest, BackwardZeroAtSaturation) {
  SigmoidLayer layer(MakeLayerParam("Sigmoid"));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 2});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  // Very negative → sigmoid ≈ 0, derivative ≈ 0
  // Very positive → sigmoid ≈ 1, derivative ≈ 0
  SetBlobData(&bottom, {-20.0f, 20.0f});
  layer.Forward(bottoms, tops);
  SetTopDiffToOnes(&top);

  std::vector<bool> prop_down = {true};
  layer.Backward(tops, prop_down, bottoms);

  const float* dx = bottom.cpu_diff();
  EXPECT_NEAR(dx[0], 0.0f, 1e-6f);
  EXPECT_NEAR(dx[1], 0.0f, 1e-6f);
}

TEST(SigmoidLayerTest, BackwardMaxAtZero) {
  SigmoidLayer layer(MakeLayerParam("Sigmoid"));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 1});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {0.0f});
  layer.Forward(bottoms, tops);
  SetTopDiffToOnes(&top);

  std::vector<bool> prop_down = {true};
  layer.Backward(tops, prop_down, bottoms);

  // At x=0, sigmoid=0.5, dy=1 → dx = 1 * 0.5 * 0.5 = 0.25
  EXPECT_NEAR(bottom.cpu_diff()[0], 0.25f, 1e-6f);
}

TEST(SigmoidLayerTest, GradientCheck) {
  SigmoidLayer layer(MakeLayerParam("Sigmoid"));
  Blob bottom(std::vector<int64_t>{1, 1, 2, 2});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  float* bdata = bottom.cpu_mutable_data();
  float vals[] = {-2.0f, -0.5f, 0.5f, 1.5f};
  std::memcpy(bdata, vals, sizeof(vals));

  auto forward_loss = [&]() {
    layer.Forward(bottoms, tops);
    float loss = 0.0f;
    const float* td = top.cpu_data();
    for (int64_t i = 0; i < top.count(); ++i) loss += td[i] * td[i] * 0.5f;
    float* tdiff = top.cpu_mutable_diff();
    for (int64_t i = 0; i < top.count(); ++i) tdiff[i] = td[i];
    return loss;
  };

  forward_loss();
  std::vector<bool> prop_down = {true};
  layer.Backward(tops, prop_down, bottoms);
  CheckGradient([&]() { return forward_loss(); }, &bottom, bottom.cpu_diff(), 4);
}

// ────────────────────────────────────────────────────────────────────────────
//  TanH tests
// ────────────────────────────────────────────────────────────────────────────

TEST(TanHLayerTest, TypeName) {
  TanHLayer layer(MakeLayerParam("TanH"));
  EXPECT_STREQ(layer.type(), "TanH");
}

TEST(TanHLayerTest, ForwardKnownValues) {
  TanHLayer layer(MakeLayerParam("TanH"));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 5});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f});
  layer.Forward(bottoms, tops);

  const float* y = top.cpu_data();
  EXPECT_NEAR(y[0], -1.0f, 1e-4f);      // tanh(-10) ≈ -1
  EXPECT_NEAR(y[1], std::tanh(-1.0f), 1e-6f);
  EXPECT_NEAR(y[2], 0.0f, 1e-6f);       // tanh(0) = 0
  EXPECT_NEAR(y[3], std::tanh(1.0f), 1e-6f);
  EXPECT_NEAR(y[4], 1.0f, 1e-4f);       // tanh(10) ≈ 1
}

TEST(TanHLayerTest, ForwardIsOdd) {
  // tanh(-x) = -tanh(x)
  TanHLayer layer(MakeLayerParam("TanH"));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 4});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-2.0f, -0.5f, 0.5f, 2.0f});
  layer.Forward(bottoms, tops);

  const float* y = top.cpu_data();
  EXPECT_NEAR(y[0], -y[3], 1e-6f);
  EXPECT_NEAR(y[1], -y[2], 1e-6f);
}

TEST(TanHLayerTest, BackwardZeroAtSaturation) {
  TanHLayer layer(MakeLayerParam("TanH"));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 2});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-20.0f, 20.0f});
  layer.Forward(bottoms, tops);
  SetTopDiffToOnes(&top);

  std::vector<bool> prop_down = {true};
  layer.Backward(tops, prop_down, bottoms);

  const float* dx = bottom.cpu_diff();
  EXPECT_NEAR(dx[0], 0.0f, 1e-4f);  // tanh' = 1-tanh^2 ≈ 0 at saturation
  EXPECT_NEAR(dx[1], 0.0f, 1e-4f);
}

TEST(TanHLayerTest, BackwardMaxAtZero) {
  TanHLayer layer(MakeLayerParam("TanH"));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 1});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {0.0f});
  layer.Forward(bottoms, tops);
  SetTopDiffToOnes(&top);

  std::vector<bool> prop_down = {true};
  layer.Backward(tops, prop_down, bottoms);

  // At x=0, tanh=0, dx = 1 * (1 - 0) = 1
  EXPECT_NEAR(bottom.cpu_diff()[0], 1.0f, 1e-6f);
}

TEST(TanHLayerTest, GradientCheck) {
  TanHLayer layer(MakeLayerParam("TanH"));
  Blob bottom(std::vector<int64_t>{1, 1, 2, 2});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  float* bdata = bottom.cpu_mutable_data();
  float vals[] = {-2.0f, -0.5f, 0.5f, 1.5f};
  std::memcpy(bdata, vals, sizeof(vals));

  auto forward_loss = [&]() {
    layer.Forward(bottoms, tops);
    float loss = 0.0f;
    const float* td = top.cpu_data();
    for (int64_t i = 0; i < top.count(); ++i) loss += td[i] * td[i] * 0.5f;
    float* tdiff = top.cpu_mutable_diff();
    for (int64_t i = 0; i < top.count(); ++i) tdiff[i] = td[i];
    return loss;
  };

  forward_loss();
  std::vector<bool> prop_down = {true};
  layer.Backward(tops, prop_down, bottoms);
  CheckGradient([&]() { return forward_loss(); }, &bottom, bottom.cpu_diff(), 4);
}

// ────────────────────────────────────────────────────────────────────────────
//  ELU tests
// ────────────────────────────────────────────────────────────────────────────

TEST(ELULayerTest, TypeName) {
  ELULayer layer(MakeELUParam());
  EXPECT_STREQ(layer.type(), "ELU");
}

TEST(ELULayerTest, ForwardPositiveLinear) {
  ELULayer layer(MakeELUParam(1.0f));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 3});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {0.0f, 1.0f, 5.0f});
  layer.Forward(bottoms, tops);

  const float* y = top.cpu_data();
  EXPECT_FLOAT_EQ(y[0], 0.0f);  // x=0 → y=0
  EXPECT_FLOAT_EQ(y[1], 1.0f);  // x=1 → y=1
  EXPECT_FLOAT_EQ(y[2], 5.0f);  // x=5 → y=5
}

TEST(ELULayerTest, ForwardNegativeExponential) {
  float alpha = 1.0f;
  ELULayer layer(MakeELUParam(alpha));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 2});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-1.0f, -2.0f});
  layer.Forward(bottoms, tops);

  const float* y = top.cpu_data();
  float expected0 = alpha * (std::exp(-1.0f) - 1.0f);
  float expected1 = alpha * (std::exp(-2.0f) - 1.0f);
  EXPECT_NEAR(y[0], expected0, 1e-5f);
  EXPECT_NEAR(y[1], expected1, 1e-5f);
  // y > -alpha for negative x
  EXPECT_GT(y[0], -alpha);
  EXPECT_GT(y[1], -alpha);
}

TEST(ELULayerTest, ForwardCustomAlpha) {
  float alpha = 0.5f;
  ELULayer layer(MakeELUParam(alpha));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 1});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-1.0f});
  layer.Forward(bottoms, tops);
  float expected = alpha * (std::exp(-1.0f) - 1.0f);
  EXPECT_NEAR(top.cpu_data()[0], expected, 1e-5f);
}

TEST(ELULayerTest, BackwardPositive) {
  ELULayer layer(MakeELUParam(1.0f));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 2});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {1.0f, 3.0f});
  layer.Forward(bottoms, tops);
  SetTopDiffToOnes(&top);

  std::vector<bool> prop_down = {true};
  layer.Backward(tops, prop_down, bottoms);

  const float* dx = bottom.cpu_diff();
  EXPECT_FLOAT_EQ(dx[0], 1.0f);  // x>0 → dx = dy = 1
  EXPECT_FLOAT_EQ(dx[1], 1.0f);
}

TEST(ELULayerTest, BackwardNegative) {
  ELULayer layer(MakeELUParam(1.0f));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 1});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-1.0f});
  layer.Forward(bottoms, tops);
  SetTopDiffToOnes(&top);

  std::vector<bool> prop_down = {true};
  layer.Backward(tops, prop_down, bottoms);

  // For x<0: dx = dy * (y + alpha)
  float y = top.cpu_data()[0];
  EXPECT_NEAR(bottom.cpu_diff()[0], 1.0f * (y + 1.0f), 1e-5f);
}

TEST(ELULayerTest, BackwardContinuityAtZero) {
  // At x=0: forward y=0, dx for positive side = 1, dx for negative side = dy*(0+alpha) = alpha
  // The gradient has a discontinuity at 0 when alpha != 1
  ELULayer layer1(MakeELUParam(1.0f));
  Blob b1(std::vector<int64_t>{1,1,1,1}), t1;
  std::vector<Blob*> bt1={&b1}, tp1={&t1};
  layer1.LayerSetUp(bt1, tp1); layer1.Reshape(bt1, tp1);
  SetBlobData(&b1, {0.0f});
  layer1.Forward(bt1, tp1); SetTopDiffToOnes(&t1);
  std::vector<bool> pd={true};
  layer1.Backward(tp1, pd, bt1);
  // alpha=1: dx = dy*(0+1) = 1, same as positive side → continuous
  EXPECT_NEAR(b1.cpu_diff()[0], 1.0f, 1e-6f);
}

TEST(ELULayerTest, GradientCheck) {
  ELULayer layer(MakeELUParam(1.0f));
  Blob bottom(std::vector<int64_t>{1, 1, 2, 2});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  float* bdata = bottom.cpu_mutable_data();
  float vals[] = {-2.0f, -0.5f, 0.5f, 2.0f};
  std::memcpy(bdata, vals, sizeof(vals));

  auto forward_loss = [&]() {
    layer.Forward(bottoms, tops);
    float loss = 0.0f;
    const float* td = top.cpu_data();
    for (int64_t i = 0; i < top.count(); ++i) loss += td[i] * td[i] * 0.5f;
    float* tdiff = top.cpu_mutable_diff();
    for (int64_t i = 0; i < top.count(); ++i) tdiff[i] = td[i];
    return loss;
  };

  forward_loss();
  std::vector<bool> prop_down = {true};
  layer.Backward(tops, prop_down, bottoms);
  CheckGradient([&]() { return forward_loss(); }, &bottom, bottom.cpu_diff(), 4);
}

// ────────────────────────────────────────────────────────────────────────────
//  PReLU tests
// ────────────────────────────────────────────────────────────────────────────

TEST(PReLULayerTest, TypeName) {
  PReLULayer layer(MakePReLUParam(true, 0.25f));
  EXPECT_STREQ(layer.type(), "PReLU");
}

TEST(PReLULayerTest, ForwardChannelShared) {
  // PReLU with channel_shared=true, default slope=0.25
  PReLULayer layer(MakePReLUParam(true));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 4});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  // slope should be initialized to 0.25
  SetBlobData(&bottom, {-2.0f, -0.5f, 0.0f, 3.0f});
  layer.Forward(bottoms, tops);

  const float* y = top.cpu_data();
  EXPECT_NEAR(y[0], -2.0f * 0.25f, 1e-6f);  // negative: slope * x
  EXPECT_NEAR(y[1], -0.5f * 0.25f, 1e-6f);
  EXPECT_NEAR(y[2], 0.0f, 1e-6f);
  EXPECT_NEAR(y[3], 3.0f, 1e-6f);            // positive: x
}

TEST(PReLULayerTest, ForwardPerChannel) {
  // PReLU with channel_shared=false, 2 channels, different slopes
  caffe::LayerParameter param = MakeLayerParam("PReLU");
  param.mutable_prelu_param()->set_channel_shared(false);
  PReLULayer layer(param);

  Blob bottom(std::vector<int64_t>{1, 2, 1, 2});  // N=1, C=2, H=1, W=2
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  // Set per-channel slopes: channel 0 slope=0.1, channel 1 slope=0.5
  auto& slope_blob = layer.blobs()[0];
  float* slopes = slope_blob->cpu_mutable_data();
  slopes[0] = 0.1f;
  slopes[1] = 0.5f;

  // Layout: [ch0_pos0, ch0_pos1, ch1_pos0, ch1_pos1]
  SetBlobData(&bottom, {-1.0f, 2.0f, -1.0f, 2.0f});
  layer.Forward(bottoms, tops);

  const float* y = top.cpu_data();
  EXPECT_NEAR(y[0], -1.0f * 0.1f, 1e-6f);  // ch0, negative → slope0*x
  EXPECT_NEAR(y[1], 2.0f, 1e-6f);            // ch0, positive → x
  EXPECT_NEAR(y[2], -1.0f * 0.5f, 1e-6f);  // ch1, negative → slope1*x
  EXPECT_NEAR(y[3], 2.0f, 1e-6f);            // ch1, positive → x
}

TEST(PReLULayerTest, BackwardChannelShared) {
  PReLULayer layer(MakePReLUParam(true));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 3});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-1.0f, 0.0f, 2.0f});
  layer.Forward(bottoms, tops);
  SetTopDiffToOnes(&top);

  std::vector<bool> prop_down = {true};
  // Enable param propagation
  layer.set_param_propagate_down(0, true);
  layer.Backward(tops, prop_down, bottoms);

  const float* dx = bottom.cpu_diff();
  // x<0 → dx = dy * slope = 1 * 0.25 = 0.25
  // x=0 → dx = dy * slope = 0.25 (since PReLU uses x>0 check, 0 goes to negative branch)
  // x>0 → dx = dy = 1
  EXPECT_NEAR(dx[0], 0.25f, 1e-6f);
  EXPECT_NEAR(dx[2], 1.0f, 1e-6f);
}

TEST(PReLULayerTest, BackwardSlopeGradient) {
  PReLULayer layer(MakePReLUParam(true));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 3});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  // Set slope to 0.25 and input with negative values
  SetBlobData(&bottom, {-2.0f, -1.0f, 3.0f});
  layer.Forward(bottoms, tops);
  SetTopDiffToOnes(&top);

  std::vector<bool> prop_down = {true};
  layer.set_param_propagate_down(0, true);
  layer.Backward(tops, prop_down, bottoms);

  // d_slope = sum over x<0 of dy * x = 1*(-2) + 1*(-1) = -3
  float d_slope = layer.blobs()[0]->cpu_diff()[0];
  EXPECT_NEAR(d_slope, -3.0f, 1e-5f);
}

TEST(PReLULayerTest, BackwardSkipPropagateDown) {
  PReLULayer layer(MakePReLUParam(true));
  Blob bottom(std::vector<int64_t>{1, 1, 1, 2});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  SetBlobData(&bottom, {-1.0f, 1.0f});
  layer.Forward(bottoms, tops);

  float sentinel = 777.0f;
  float* bdiff = bottom.cpu_mutable_diff();
  for (int64_t i = 0; i < bottom.count(); ++i) bdiff[i] = sentinel;

  std::vector<bool> prop_down = {false};
  layer.set_param_propagate_down(0, false);
  layer.Backward(tops, prop_down, bottoms);

  for (int64_t i = 0; i < bottom.count(); ++i) {
    EXPECT_FLOAT_EQ(bdiff[i], sentinel);
  }
}

TEST(PReLULayerTest, GradientCheck) {
  PReLULayer layer(MakePReLUParam(true));
  Blob bottom(std::vector<int64_t>{1, 1, 2, 2});
  Blob top;
  std::vector<Blob*> bottoms = {&bottom};
  std::vector<Blob*> tops = {&top};
  layer.LayerSetUp(bottoms, tops);
  layer.Reshape(bottoms, tops);

  // Set slope to 0.3 for testing
  layer.blobs()[0]->cpu_mutable_data()[0] = 0.3f;

  float* bdata = bottom.cpu_mutable_data();
  float vals[] = {-1.5f, -0.3f, 0.3f, 1.5f};
  std::memcpy(bdata, vals, sizeof(vals));

  auto forward_loss = [&]() {
    layer.Forward(bottoms, tops);
    float loss = 0.0f;
    const float* td = top.cpu_data();
    for (int64_t i = 0; i < top.count(); ++i) loss += td[i] * td[i] * 0.5f;
    float* tdiff = top.cpu_mutable_diff();
    for (int64_t i = 0; i < top.count(); ++i) tdiff[i] = td[i];
    return loss;
  };

  forward_loss();
  std::vector<bool> prop_down = {true};
  layer.set_param_propagate_down(0, false);  // don't compute weight grad for input grad check
  layer.Backward(tops, prop_down, bottoms);
  CheckGradient([&]() { return forward_loss(); }, &bottom, bottom.cpu_diff(), 4);
}

// ────────────────────────────────────────────────────────────────────────────
//  Cross-layer: NeuronLayer Inheritance verification
// ────────────────────────────────────────────────────────────────────────────

TEST(NeuronLayerTest, AllActivationsPreserveShape) {
  // All 5 activation layers should produce output of same shape as input
  std::vector<int64_t> shapes[] = {{8}, {2, 6}, {1, 3, 4}, {2, 4, 3, 5}};

  auto test_layer = [&](Layer* layer, const std::vector<int64_t>& shape) {
    Blob bottom(shape);
    Blob top;
    std::vector<Blob*> bottoms = {&bottom};
    std::vector<Blob*> tops = {&top};
    layer->LayerSetUp(bottoms, tops);
    layer->Reshape(bottoms, tops);
    EXPECT_EQ(top.count(), bottom.count());
    for (int a = 0; a < static_cast<int>(shape.size()); ++a) {
      EXPECT_EQ(top.shape(a), shape[a]) << "Shape mismatch at axis " << a;
    }
  };

  for (auto& shape : shapes) {
    ReLULayer relu(MakeReLUParam());
    SigmoidLayer sig(MakeLayerParam("Sigmoid"));
    TanHLayer tanh_l(MakeLayerParam("TanH"));
    ELULayer elu(MakeELUParam());
    PReLULayer prelu(MakePReLUParam(true));

    test_layer(&relu, shape);
    test_layer(&sig, shape);
    test_layer(&tanh_l, shape);
    test_layer(&elu, shape);
    test_layer(&prelu, shape);
  }
}

TEST(NeuronLayerTest, AllActivationsHaveCorrectBlobCounts) {
  // ReLU, Sigmoid, TanH, ELU have 0 learnable blobs; PReLU has 1 (slope)
  {
    ReLULayer layer(MakeReLUParam());
    Blob b(std::vector<int64_t>{1,1,1,1}), t;
    std::vector<Blob*> bt={&b}, tp={&t};
    layer.LayerSetUp(bt, tp);
    layer.Reshape(bt, tp);
    EXPECT_EQ(layer.blobs().size(), 0U);
  }
  {
    SigmoidLayer layer(MakeLayerParam("Sigmoid"));
    Blob b(std::vector<int64_t>{1,1,1,1}), t;
    std::vector<Blob*> bt={&b}, tp={&t};
    layer.LayerSetUp(bt, tp);
    layer.Reshape(bt, tp);
    EXPECT_EQ(layer.blobs().size(), 0U);
  }
  {
    TanHLayer layer(MakeLayerParam("TanH"));
    Blob b(std::vector<int64_t>{1,1,1,1}), t;
    std::vector<Blob*> bt={&b}, tp={&t};
    layer.LayerSetUp(bt, tp);
    layer.Reshape(bt, tp);
    EXPECT_EQ(layer.blobs().size(), 0U);
  }
  {
    ELULayer layer(MakeELUParam());
    Blob b(std::vector<int64_t>{1,1,1,1}), t;
    std::vector<Blob*> bt={&b}, tp={&t};
    layer.LayerSetUp(bt, tp);
    layer.Reshape(bt, tp);
    EXPECT_EQ(layer.blobs().size(), 0U);
  }
  {
    PReLULayer layer(MakePReLUParam(true));
    Blob b(std::vector<int64_t>{1,1,1,1}), t;
    std::vector<Blob*> bt={&b}, tp={&t};
    layer.LayerSetUp(bt, tp);
    layer.Reshape(bt, tp);
    EXPECT_EQ(layer.blobs().size(), 1U);
  }
}
