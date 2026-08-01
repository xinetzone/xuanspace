#include "test_harness.hpp"
#include "caffe_ffi/net.hpp"
#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include <string>
#include <vector>
#include <cstdint>

using namespace caffe_ffi;

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

// Count auto-inserted Split layers. Uses a two-check strategy:
// 1. Name pattern: ends with "_split" (native Caffe naming: *_<producer>_<idx>_split)
// 2. Type check: layer type() must be "Split" (prevents false positives from
//    user-defined layers whose name happens to end with "_split")
static int CountAutoSplitLayers(ObjectPtr<Net> net) {
  int count = 0;
  for (const auto& name : net->layer_names()) {
    // Auto-inserted splits follow: <blob>_<producer>_<idx>_split
    // i.e. they END with "_split" and are Split layers
    if (name.size() >= 7) {
      size_t pos = name.rfind("_split");
      if (pos != std::string::npos && pos == name.size() - 6) {
        // Double-check: it must be a Split type layer
        ObjectPtr<Layer> layer;
        try { layer = net->layer_by_name(name); } catch (...) { continue; }
        if (layer && std::string(layer->type()) == "Split") {
          count++;
        }
      }
    }
  }
  return count;
}

static bool HasLayerNamed(ObjectPtr<Net> net, const std::string& name) {
  for (const auto& n : net->layer_names()) {
    if (n == name) return true;
  }
  return false;
}

static int LayerPosition(ObjectPtr<Net> net, const std::string& name) {
  const auto& names = net->layer_names();
  for (int i = 0; i < static_cast<int>(names.size()); ++i) {
    if (names[i] == name) return i;
  }
  return -1;
}

// ──────────────────────────────────────────────────────────────────────
// Prototxt definitions
// ──────────────────────────────────────────────────────────────────────

// Two consumers sharing the external input "data" → 1 auto split for "data"
static const char* kTwoConsumerProto = R"(
name: "TwoConsumerNet"
input: "data"
input_shape { dim: 2 dim: 4 }
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "data"
  top: "fc1_out"
  inner_product_param { num_output: 3 bias_term: true
    weight_filler { type: "constant" value: 1.0 }
    bias_filler { type: "constant" value: 0.0 }
  }
}
layer {
  name: "fc2"
  type: "InnerProduct"
  bottom: "data"
  top: "fc2_out"
  inner_product_param { num_output: 3 bias_term: true
    weight_filler { type: "constant" value: 1.0 }
    bias_filler { type: "constant" value: 0.0 }
  }
}
)";

// In-place ReLU followed by two consumers.
// data → fc1 (1 consumer: relu1) → relu1 (in-place, fc1_out, 2 consumers: fc2/fc3)
// → only fc1_out (after relu1) needs a split; data has 1 consumer so no split.
static const char* kInplaceTwoConsumerProto = R"(
name: "InplaceNet"
input: "data"
input_shape { dim: 2 dim: 4 }
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "data"
  top: "fc1_out"
  inner_product_param { num_output: 3 bias_term: false
    weight_filler { type: "constant" value: 1.0 }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "fc1_out"
  top: "fc1_out"
}
layer {
  name: "fc2"
  type: "InnerProduct"
  bottom: "fc1_out"
  top: "fc2_out"
  inner_product_param { num_output: 3 bias_term: false
    weight_filler { type: "constant" value: 1.0 }
  }
}
layer {
  name: "fc3"
  type: "InnerProduct"
  bottom: "fc1_out"
  top: "fc3_out"
  inner_product_param { num_output: 3 bias_term: false
    weight_filler { type: "constant" value: 1.0 }
  }
}
)";

// Two consumers AND data feeds two branches -- tests both external input split
// AND in-place split simultaneously.
static const char* kDataAndInplaceSplitProto = R"(
name: "DataAndInplaceNet"
input: "data"
input_shape { dim: 2 dim: 4 }
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "data"
  top: "fc1_out"
  inner_product_param { num_output: 3 bias_term: false
    weight_filler { type: "constant" value: 1.0 }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "fc1_out"
  top: "fc1_out"
}
layer {
  name: "fc2"
  type: "InnerProduct"
  bottom: "fc1_out"
  top: "fc2_out"
  inner_product_param { num_output: 3 bias_term: false
    weight_filler { type: "constant" value: 1.0 }
  }
}
layer {
  name: "fc3"
  type: "InnerProduct"
  bottom: "fc1_out"
  top: "fc3_out"
  inner_product_param { num_output: 3 bias_term: false
    weight_filler { type: "constant" value: 1.0 }
  }
}
layer {
  name: "fc_branch"
  type: "InnerProduct"
  bottom: "data"
  top: "branch_out"
  inner_product_param { num_output: 3 bias_term: false
    weight_filler { type: "constant" value: 1.0 }
  }
}
)";

// Linear chain (no fan-out) -- zero splits expected
static const char* kLinearChainProto = R"(
name: "LinearNet"
input: "data"
input_shape { dim: 2 dim: 4 }
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "data"
  top: "fc1_out"
  inner_product_param { num_output: 3 }
}
layer {
  name: "relu"
  type: "ReLU"
  bottom: "fc1_out"
  top: "fc1_out"
}
layer {
  name: "fc2"
  type: "InnerProduct"
  bottom: "fc1_out"
  top: "fc2_out"
  inner_product_param { num_output: 3 }
}
)";

// Single-consumer blob -- zero splits
static const char* kSingleConsumerProto = R"(
name: "SingleNet"
input: "data"
input_shape { dim: 1 dim: 4 }
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "data"
  top: "out"
  inner_product_param { num_output: 3 }
}
)";

// Input layer (modern style) with 3 consumers -- 1 split after Input layer
static const char* kInputLayerThreeConsumerProto = R"(
name: "InputLayerNet"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 4 } }
}
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "data"
  top: "fc1_out"
  inner_product_param { num_output: 3 }
}
layer {
  name: "fc2"
  type: "InnerProduct"
  bottom: "data"
  top: "fc2_out"
  inner_product_param { num_output: 3 }
}
layer {
  name: "fc3"
  type: "InnerProduct"
  bottom: "data"
  top: "fc3_out"
  inner_product_param { num_output: 3 }
}
)";

// Explicit Split layer (idempotency test) -- no additional splits added
static const char* kExplicitSplitProto = R"(
name: "ExplicitSplitNet"
input: "data"
input_shape { dim: 1 dim: 4 }
layer {
  name: "data_input_0_split"
  type: "Split"
  bottom: "data"
  top: "data_input_0_split_0"
  top: "data_input_0_split_1"
}
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "data_input_0_split_0"
  top: "out1"
  inner_product_param { num_output: 3 }
}
layer {
  name: "fc2"
  type: "InnerProduct"
  bottom: "data_input_0_split_1"
  top: "out2"
  inner_product_param { num_output: 3 }
}
)";

// Loss weight on intermediate blob counts as a consumer
static const char* kLossWeightProto = R"(
name: "LossNet"
force_backward: true
input: "data"
input_shape { dim: 2 dim: 4 }
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "data"
  top: "fc1_out"
  inner_product_param { num_output: 3 bias_term: false
    weight_filler { type: "constant" value: 1.0 }
  }
  loss_weight: 1.0
}
layer {
  name: "fc2"
  type: "InnerProduct"
  bottom: "fc1_out"
  top: "fc2_out"
  inner_product_param { num_output: 3 bias_term: false
    weight_filler { type: "constant" value: 1.0 }
  }
}
layer {
  name: "fc3"
  type: "InnerProduct"
  bottom: "fc1_out"
  top: "fc3_out"
  inner_product_param { num_output: 3 bias_term: false
    weight_filler { type: "constant" value: 1.0 }
  }
}
)";

// Empty network (zero layers)
static const char* kEmptyProto = R"(
name: "EmptyNet"
input: "data"
input_shape { dim: 1 dim: 4 }
)";

// Unknown bottom blob -- should throw
static const char* kBadRefProto = R"(
name: "BadNet"
input: "data"
input_shape { dim: 1 dim: 4 }
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "nonexistent"
  top: "out"
  inner_product_param { num_output: 3 }
}
)";

// Double in-place: fc→relu→relu with two consumers after the second relu.
// data → fc1 (1 consumer) → relu1 (in-place, 1 consumer: relu2) →
// relu2 (in-place, 2 consumers: fc2/fc3) → only x after relu2 needs split.
static const char* kDoubleInplaceProto = R"(
name: "DoubleInplaceNet"
input: "data"
input_shape { dim: 2 dim: 4 }
layer {
  name: "fc1"
  type: "InnerProduct"
  bottom: "data"
  top: "x"
  inner_product_param { num_output: 3 }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "x"
  top: "x"
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "x"
  top: "x"
}
layer {
  name: "fc2"
  type: "InnerProduct"
  bottom: "x"
  top: "fc2_out"
  inner_product_param { num_output: 3 }
}
layer {
  name: "fc3"
  type: "InnerProduct"
  bottom: "x"
  top: "fc3_out"
  inner_product_param { num_output: 3 }
}
)";

// ──────────────────────────────────────────────────────────────────────
// Helper to fill blob with constant value
// ──────────────────────────────────────────────────────────────────────
static void FillBlobConstant(ObjectPtr<Blob> blob, float value) {
  float* data = blob->cpu_mutable_data();
  int64_t n = blob->count();
  for (int64_t i = 0; i < n; ++i) data[i] = value;
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────

// 1. Basic split insertion for multi-consumer external input
TEST(InsertSplitsTest, TwoConsumersAutoInsertsSplit) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kTwoConsumerProto));
  // DAG: data(external) → fc1 + fc2 = 2 consumers → split for data
  EXPECT_EQ(CountAutoSplitLayers(net), 1);
  EXPECT_TRUE(HasLayerNamed(net, "data_input_0_split"));
}

// 2. External input split must be at position 0
TEST(InsertSplitsTest, ExternalInputSplitAtPositionZero) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kTwoConsumerProto));
  int pos = LayerPosition(net, "data_input_0_split");
  EXPECT_EQ(pos, 0);
}

// 3. In-place ReLU: split named after last in-place producer (relu1), not fc1
TEST(InsertSplitsTest, InplaceSplitNamedAfterLastProducer) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kInplaceTwoConsumerProto));
  // data has 1 consumer (fc1) → no split for data; fc1_out after relu1 has 2 → 1 split
  EXPECT_EQ(CountAutoSplitLayers(net), 1);
  EXPECT_TRUE(HasLayerNamed(net, "fc1_out_relu1_0_split"));
  // No split after fc1 (only 1 consumer: relu1)
  EXPECT_FALSE(HasLayerNamed(net, "fc1_out_fc1_0_split"));
}

// 4. In-place split positioned immediately after the last in-place producer
TEST(InsertSplitsTest, InplaceSplitPositionedAfterProducer) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kInplaceTwoConsumerProto));
  int relu_pos = LayerPosition(net, "relu1");
  int split_pos = LayerPosition(net, "fc1_out_relu1_0_split");
  EXPECT_EQ(split_pos, relu_pos + 1);
}

// 5. Linear chain (no fan-out) produces zero splits
TEST(InsertSplitsTest, LinearChainZeroSplits) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kLinearChainProto));
  // DAG: data → fc1 → fc2 → fc3: every blob has exactly 1 consumer → 0 splits
  EXPECT_EQ(CountAutoSplitLayers(net), 0);
  EXPECT_EQ(net->layer_names().size(), static_cast<size_t>(3));
}

// 6. Single consumer produces zero splits
TEST(InsertSplitsTest, SingleConsumerZeroSplits) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kSingleConsumerProto));
  // DAG: data → fc1 → fc2: every blob has exactly 1 consumer → 0 splits
  EXPECT_EQ(CountAutoSplitLayers(net), 0);
}

// 7. Explicit Input layer with 3 consumers: split named after data layer
TEST(InsertSplitsTest, InputLayerThreeConsumers) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kInputLayerThreeConsumerProto));
  // DAG: data(type=Input) → fc1 + fc2 + fc3 = 3 consumers → 1 split (named after "data" layer)
  EXPECT_EQ(CountAutoSplitLayers(net), 1);
  EXPECT_TRUE(HasLayerNamed(net, "data_data_0_split"));
  int data_pos = LayerPosition(net, "data");
  int split_pos = LayerPosition(net, "data_data_0_split");
  EXPECT_EQ(split_pos, data_pos + 1);
}

// 8. Idempotency: network with explicit Split does not get duplicate splits
TEST(InsertSplitsTest, IdempotentNoDuplicateSplits) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kExplicitSplitProto));
  // DAG: data → explicit_split(type=Split) → fc1 + fc2
  //   data has 1 consumer (explicit split) → no new external input split
  //   split_0/split_1 each have 1 consumer (fc1/fc2) → no additional splits
  //   → 0 NEW splits inserted; explicit split preserved (total 3 layers)
  EXPECT_EQ(CountAutoSplitLayers(net), 1);
  EXPECT_TRUE(HasLayerNamed(net, "data_input_0_split"));
  EXPECT_EQ(net->layer_names().size(), static_cast<size_t>(3));
}

// 9. Loss weight triggers split (loss output counts as a consumer)
TEST(InsertSplitsTest, LossWeightTriggersSplit) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kLossWeightProto));
  // data → fc1 only (1 consumer, no split); fc1_out → fc2 + fc3 + loss = 3 consumers
  EXPECT_EQ(CountAutoSplitLayers(net), 1);
  EXPECT_TRUE(HasLayerNamed(net, "fc1_out_fc1_0_split"));
}

// 10. Empty network doesn't crash
TEST(InsertSplitsTest, EmptyNetworkNoCrash) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kEmptyProto));
  EXPECT_EQ(CountAutoSplitLayers(net), 0);
  EXPECT_EQ(net->layer_names().size(), static_cast<size_t>(0));
}

// 11. Unknown bottom blob raises error
TEST(InsertSplitsTest, UnknownBottomRaisesError) {
  bool threw = false;
  try {
    make_object<Net>(ReadNetParamsFromTextString(kBadRefProto));
  } catch (const std::exception& e) {
    threw = true;
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("nonexistent") != std::string::npos ||
                msg.find("Unknown bottom blob") != std::string::npos);
  }
  EXPECT_TRUE(threw);
}

// 12. Double in-place: split named after last producer (relu2)
TEST(InsertSplitsTest, DoubleInplaceSplitAfterLastProducer) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kDoubleInplaceProto));
  // data → fc1 (1 consumer), fc1→relu1 (1 consumer), relu1→relu2 (in-place, 2 consumers: fc2/fc3)
  EXPECT_EQ(CountAutoSplitLayers(net), 1);
  EXPECT_TRUE(HasLayerNamed(net, "x_relu2_0_split"));
  EXPECT_FALSE(HasLayerNamed(net, "x_relu1_0_split"));
  EXPECT_FALSE(HasLayerNamed(net, "x_fc1_0_split"));
}

// 13. Both external input split AND in-place split coexist
TEST(InsertSplitsTest, BothExternalInputAndInplaceSplits) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kDataAndInplaceSplitProto));
  // data → fc1 + fc_branch = 2 consumers → data_input_0_split
  // fc1_out after relu1 → fc2 + fc3 = 2 consumers → fc1_out_relu1_0_split
  EXPECT_EQ(CountAutoSplitLayers(net), 2);
  EXPECT_TRUE(HasLayerNamed(net, "data_input_0_split"));
  EXPECT_TRUE(HasLayerNamed(net, "fc1_out_relu1_0_split"));
}

// 14. Forward correctness: two-consumer net with constant weights produces correct values
TEST(InsertSplitsTest, ForwardCorrectnessTwoConsumer) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kTwoConsumerProto));
  FillBlobConstant(net->blob_by_name("data"), 1.0f);
  net->Forward({});

  // weight=1.0, bias=0.0, input=1.0 (4 dims), num_output=3
  // Each output neuron = sum_i(weight_i * input_i) + bias = 4*1.0*1.0 + 0 = 4.0
  ObjectPtr<Blob> fc1_out = net->blob_by_name("fc1_out");
  ObjectPtr<Blob> fc2_out = net->blob_by_name("fc2_out");
  EXPECT_EQ(fc1_out->num_axes(), 2);
  EXPECT_EQ(fc1_out->shape(0), 2);
  EXPECT_EQ(fc1_out->shape(1), 3);
  EXPECT_EQ(fc2_out->shape(0), 2);
  EXPECT_EQ(fc2_out->shape(1), 3);

  const float* fc1_data = fc1_out->cpu_data();
  const float* fc2_data = fc2_out->cpu_data();
  for (int64_t i = 0; i < fc1_out->count(); ++i) {
    EXPECT_NEAR(fc1_data[i], 4.0f, 1e-4f);
    EXPECT_NEAR(fc2_data[i], 4.0f, 1e-4f);
  }
}

// 15. Forward correctness: in-place ReLU + split produces correct values
TEST(InsertSplitsTest, ForwardCorrectnessInplaceSplit) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kInplaceTwoConsumerProto));
  FillBlobConstant(net->blob_by_name("data"), 1.0f);
  net->Forward({});

  // fc1: input=1.0 (4 dims), weight=1.0, bias=0, num_output=3 → fc1_out = 4.0
  // relu1: 4.0 > 0 → unchanged (4.0)
  // split: copies 4.0 to split outputs
  // fc2/fc3: input=4.0 (3 dims), weight=1.0, bias=0, num_output=3
  //         each output = 3*4.0*1.0 = 12.0
  ObjectPtr<Blob> fc2_out = net->blob_by_name("fc2_out");
  ObjectPtr<Blob> fc3_out = net->blob_by_name("fc3_out");
  const float* fc2_data = fc2_out->cpu_data();
  const float* fc3_data = fc3_out->cpu_data();
  for (int64_t i = 0; i < fc2_out->count(); ++i) {
    EXPECT_NEAR(fc2_data[i], 12.0f, 1e-3f);
    EXPECT_NEAR(fc3_data[i], 12.0f, 1e-3f);
  }
}

// 16. Forward shapes correct for 3-consumer Input layer net
TEST(InsertSplitsTest, ForwardThreeConsumersCorrectShapes) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kInputLayerThreeConsumerProto));
  FillBlobConstant(net->blob_by_name("data"), 1.0f);
  net->Forward({});

  for (const char* name : {"fc1_out", "fc2_out", "fc3_out"}) {
    ObjectPtr<Blob> b = net->blob_by_name(name);
    EXPECT_EQ(b->shape(0), 2);
    EXPECT_EQ(b->shape(1), 3);
  }
}

// 17. Naming convention matches native Caffe exactly
TEST(InsertSplitsTest, NativeCaffeNamingConvention) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kDataAndInplaceSplitProto));
  // External input: producer="input", blob="data", idx=0 → "data_input_0_split"
  EXPECT_TRUE(HasLayerNamed(net, "data_input_0_split"));
  // In-place after relu1: producer="relu1", blob="fc1_out", idx=0 → "fc1_out_relu1_0_split"
  EXPECT_TRUE(HasLayerNamed(net, "fc1_out_relu1_0_split"));
  // Split output blobs also follow naming: <blob>_<producer>_<idx>_split_<k>
  EXPECT_TRUE(net->has_blob("data_input_0_split_0"));
  EXPECT_TRUE(net->has_blob("data_input_0_split_1"));
  EXPECT_TRUE(net->has_blob("fc1_out_relu1_0_split_0"));
  EXPECT_TRUE(net->has_blob("fc1_out_relu1_0_split_1"));
}

// 18. Total layer count correct (original N layers + auto splits)
TEST(InsertSplitsTest, TotalLayerCountCorrect) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kTwoConsumerProto));
  // Original: 2 layers (fc1, fc2) + 1 auto split = 3
  EXPECT_EQ(net->layers_array().size(), static_cast<size_t>(3));
}

// 19. Input/output blob counts correct
TEST(InsertSplitsTest, InputOutputBlobCountCorrect) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kTwoConsumerProto));
  EXPECT_EQ(net->num_inputs(), 1);
  EXPECT_EQ(net->input_blob_names()[0], std::string("data"));
  EXPECT_EQ(net->num_outputs(), 2);
}

// 20. Original blob names preserved after InsertSplits
TEST(InsertSplitsTest, HasBlobOriginalNamesPreserved) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kTwoConsumerProto));
  EXPECT_TRUE(net->has_blob("data"));
  EXPECT_TRUE(net->has_blob("fc1_out"));
  EXPECT_TRUE(net->has_blob("fc2_out"));
}

// 21. Split output blob names exist
TEST(InsertSplitsTest, SplitBlobNamesExist) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kTwoConsumerProto));
  EXPECT_TRUE(net->has_blob("data_input_0_split_0"));
  EXPECT_TRUE(net->has_blob("data_input_0_split_1"));
}

// 22. Auto-inserted layers are of type "Split"
TEST(InsertSplitsTest, LayerTypesCorrect) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kTwoConsumerProto));
  ObjectPtr<Layer> split_layer = net->layer_by_name("data_input_0_split");
  EXPECT_TRUE(split_layer != nullptr);
  EXPECT_EQ(std::string(split_layer->type()), std::string("Split"));
}

// 23. In-place net forward: split outputs feed to both consumers, both see identical data
TEST(InsertSplitsTest, InplaceSplitConsumersSeeSameData) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kInplaceTwoConsumerProto));
  FillBlobConstant(net->blob_by_name("data"), 1.0f);
  net->Forward({});

  ObjectPtr<Blob> fc2_out = net->blob_by_name("fc2_out");
  ObjectPtr<Blob> fc3_out = net->blob_by_name("fc3_out");
  const float* d2 = fc2_out->cpu_data();
  const float* d3 = fc3_out->cpu_data();
  // Both consumers must see identical data (split copies correctly)
  for (int64_t i = 0; i < fc2_out->count(); ++i) {
    EXPECT_NEAR(d2[i], d3[i], 1e-6f);
  }
}

// 24. Loss weight split has 3 outputs (2 consumers + 1 loss)
TEST(InsertSplitsTest, LossWeightSplitHasCorrectOutputCount) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kLossWeightProto));
  ObjectPtr<Layer> split = net->layer_by_name("fc1_out_fc1_0_split");
  EXPECT_TRUE(split != nullptr);
  // Split layer should have 3 tops (for fc2, fc3, and the loss path)
  const caffe::LayerParameter& lp = split->layer_param();
  EXPECT_EQ(lp.top_size(), 3);
}
