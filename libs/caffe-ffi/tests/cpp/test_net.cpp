#include "test_harness.hpp"
#include "caffe_ffi/net.hpp"
#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/common.hpp"
#include <cstring>
#include <string>

using namespace caffe_ffi;

static const char* kSimpleInputProto = R"(
name: "InputNet"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 3 } }
}
)";

static const char* kTwoLayerProto = R"(
name: "TwoLayerNet"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 2 } }
}
layer {
  name: "relu"
  type: "ReLU"
  bottom: "data"
  top: "data"
}
)";

static const char* kMlpProto = R"(
name: "MlpNet"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 3 } }
}
layer {
  name: "fc"
  type: "InnerProduct"
  bottom: "data"
  top: "fc"
  inner_product_param {
    num_output: 4
    bias_term: true
    weight_filler { type: "xavier" }
    bias_filler { type: "constant" value: 0.1 }
  }
}
layer {
  name: "relu"
  type: "ReLU"
  bottom: "fc"
  top: "fc"
}
)";

TEST(NetTest, RegistryFromExe) {
  auto types = LayerRegistry::LayerTypeList();
  EXPECT_TRUE(types.size() >= 10);
}

TEST(NetTest, CreateFromProtoString) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kSimpleInputProto));
  EXPECT_TRUE(net != nullptr);
  EXPECT_EQ(net->name(), std::string("InputNet"));
}

TEST(NetTest, LayerCount) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kTwoLayerProto));
  EXPECT_EQ(net->layers_array().size(), 2);
  EXPECT_EQ(net->layer_names().size(), 2);
}

TEST(NetTest, BlobCount) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kSimpleInputProto));
  EXPECT_EQ(net->blobs_array().size(), 1);
  EXPECT_EQ(net->blob_names().size(), 1);
}

TEST(NetTest, InputOutputBlobs) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kSimpleInputProto));
  EXPECT_EQ(net->num_inputs(), 1);
  EXPECT_EQ(net->num_outputs(), 1);
  EXPECT_EQ(net->input_blob_names().size(), 1);
  EXPECT_EQ(net->output_blob_names().size(), 1);
  EXPECT_EQ(net->input_blob_names()[0], std::string("data"));
  EXPECT_EQ(net->output_blob_names()[0], std::string("data"));
}

TEST(NetTest, HasBlob) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kSimpleInputProto));
  EXPECT_TRUE(net->has_blob("data"));
  EXPECT_FALSE(net->has_blob("nonexistent"));
}

TEST(NetTest, HasLayer) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kSimpleInputProto));
  EXPECT_TRUE(net->has_layer("data"));
  EXPECT_FALSE(net->has_layer("nonexistent"));
}

TEST(NetTest, BlobByName) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kSimpleInputProto));
  ObjectPtr<Blob> blob = net->blob_by_name("data");
  EXPECT_TRUE(blob != nullptr);
  EXPECT_EQ(blob->name(), std::string("data"));
  EXPECT_EQ(blob->num_axes(), 2);
  EXPECT_EQ(blob->shape(0), 2);
  EXPECT_EQ(blob->shape(1), 3);
}

TEST(NetTest, LayerByName) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kSimpleInputProto));
  ObjectPtr<Layer> layer = net->layer_by_name("data");
  EXPECT_TRUE(layer != nullptr);
  EXPECT_EQ(std::string(layer->type()), std::string("Input"));
}

TEST(NetTest, UnknownLayerTypeThrows) {
  static const char* badProto = R"(
name: "BadNet"
layer { name: "bad" type: "NonexistentLayer" top: "bad" }
)";
  bool threw = false;
  try {
    make_object<Net>(ReadNetParamsFromTextString(badProto));
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(NetTest, ForwardSingleInput) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kTwoLayerProto));
  ObjectPtr<Blob> blob = net->blob_by_name("data");
  EXPECT_TRUE(blob != nullptr);
  blob->Reshape(ShapeView({1, 2}));
  float* data = blob->cpu_data();
  data[0] = -1.0f;
  data[1] = 2.0f;
  Map<String, ObjectPtr<Blob>> outputs = net->Forward({});
  EXPECT_TRUE(outputs.size() > 0);
  ObjectPtr<Blob> output = net->blob_by_name("data");
  const float* out = output->cpu_data();
  EXPECT_NEAR(out[0], 0.0f, 1e-6f);
  EXPECT_NEAR(out[1], 2.0f, 1e-6f);
}

TEST(NetTest, MlpNetCreation) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kMlpProto));
  EXPECT_EQ(net->name(), std::string("MlpNet"));
  EXPECT_EQ(net->layers_array().size(), 3);
  EXPECT_EQ(net->num_inputs(), 1);
  EXPECT_TRUE(net->has_layer("fc"));
  EXPECT_TRUE(net->has_layer("relu"));
  ObjectPtr<Layer> fc = net->layer_by_name("fc");
  EXPECT_TRUE(fc != nullptr);
  EXPECT_EQ(std::string(fc->type()), std::string("InnerProduct"));
}

TEST(NetTest, LayerBlobsExistForInnerProduct) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kMlpProto));
  ObjectPtr<Layer> fc = net->layer_by_name("fc");
  EXPECT_TRUE(fc != nullptr);
  auto& blobs = fc->blobs();
  EXPECT_GE(blobs.size(), 1);
}

// ---- Error handling tests ----

TEST(NetTest, BlobByNameNotFoundThrows) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kSimpleInputProto));
  bool threw = false;
  try {
    net->blob_by_name("nonexistent_blob");
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(NetTest, LayerByNameNotFoundThrows) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kSimpleInputProto));
  bool threw = false;
  try {
    net->layer_by_name("nonexistent_layer");
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(NetTest, LayerBlobsArrayViaReflection) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kMlpProto));
  ObjectPtr<Layer> fc = net->layer_by_name("fc");
  EXPECT_TRUE(fc != nullptr);
  Array<ObjectPtr<Blob>> blobs = fc->blobs_array();
  EXPECT_GE(blobs.size(), 1);
}

TEST(NetTest, NetBlobsArrayViaReflection) {
  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(kTwoLayerProto));
  Array<ObjectPtr<Blob>> blobs = net->blobs_array();
  EXPECT_EQ(blobs.size(), 1);
  Array<ObjectPtr<Layer>> layers = net->layers_array();
  EXPECT_EQ(layers.size(), 2);
}
