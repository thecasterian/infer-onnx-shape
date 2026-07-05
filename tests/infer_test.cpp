#include <gtest/gtest.h>

#include "infer.h"
#include "model_builders.h"

// Structural inference: an intermediate tensor gets a value_info shape.
TEST(Inference, ReluChainAnnotatesIntermediate) {
  onnx::GraphProto g;
  ios::test::add_input(&g, "X", onnx::TensorProto::FLOAT, {2, 3});
  ios::test::add_node(&g, "Relu", {"X"}, {"H"});
  ios::test::add_node(&g, "Relu", {"H"}, {"Y"});
  ios::test::add_output(&g, "Y");
  auto model = ios::test::make_model(g);

  auto r = ios::infer_shapes(model, /*strict=*/false);

  auto h = ios::test::get_shape(model.graph(), "H");
  ASSERT_TRUE(h.has_value());
  EXPECT_EQ(*h, (std::vector<int64_t>{2, 3}));
  EXPECT_GE(r.annotated, 1);
  EXPECT_EQ(r.total_intermediate, 1);
}

// Data propagation: Reshape output resolved from an inline shape initializer.
TEST(Inference, ReshapeResolvesFromInlineInitializer) {
  onnx::GraphProto g;
  ios::test::add_input(&g, "X", onnx::TensorProto::FLOAT, {2, 3, 4});
  ios::test::add_int64_initializer(&g, "newshape", {2}, {6, 4});
  ios::test::add_node(&g, "Reshape", {"X", "newshape"}, {"H"});
  ios::test::add_node(&g, "Relu", {"H"}, {"Y"});
  ios::test::add_output(&g, "Y");
  auto model = ios::test::make_model(g);

  ios::infer_shapes(model, /*strict=*/false);

  auto h = ios::test::get_shape(model.graph(), "H");
  ASSERT_TRUE(h.has_value());
  EXPECT_EQ(*h, (std::vector<int64_t>{6, 4}));
}

// Data propagation is genuinely required: the Reshape shape operand is
// computed by an upstream Shape node, not read from an initializer.
TEST(Inference, ReshapeResolvesFromComputedShape) {
  onnx::GraphProto g;
  // Template tensor whose shape [6,4] we recover at runtime via Shape.
  ios::test::add_input(&g, "T", onnx::TensorProto::FLOAT, {6, 4});
  // Data tensor to be reshaped (24 elements).
  ios::test::add_input(&g, "D", onnx::TensorProto::FLOAT, {24});
  ios::test::add_node(&g, "Shape", {"T"}, {"shp"});          // shp = [6,4], computed
  ios::test::add_node(&g, "Reshape", {"D", "shp"}, {"H"});   // needs data prop to know shp
  ios::test::add_node(&g, "Relu", {"H"}, {"Y"});
  ios::test::add_output(&g, "Y");
  auto model = ios::test::make_model(g);

  ios::infer_shapes(model, /*strict=*/false);

  auto h = ios::test::get_shape(model.graph(), "H");
  ASSERT_TRUE(h.has_value());
  EXPECT_EQ(*h, (std::vector<int64_t>{6, 4}));
}

// A non-broadcastable Add is a shape-inference error: strict mode must throw,
// lenient mode must tolerate it.
TEST(Inference, StrictModeThrowsOnInferenceError) {
  onnx::GraphProto g;
  ios::test::add_input(&g, "A", onnx::TensorProto::FLOAT, {2, 3});
  ios::test::add_input(&g, "B", onnx::TensorProto::FLOAT, {4, 5});
  ios::test::add_node(&g, "Add", {"A", "B"}, {"Y"});
  ios::test::add_output(&g, "Y");
  auto model = ios::test::make_model(g);

  EXPECT_THROW(ios::infer_shapes(model, /*strict=*/true), std::runtime_error);

  onnx::GraphProto g2;
  ios::test::add_input(&g2, "A", onnx::TensorProto::FLOAT, {2, 3});
  ios::test::add_input(&g2, "B", onnx::TensorProto::FLOAT, {4, 5});
  ios::test::add_node(&g2, "Add", {"A", "B"}, {"Y"});
  ios::test::add_output(&g2, "Y");
  auto model2 = ios::test::make_model(g2);
  EXPECT_NO_THROW(ios::infer_shapes(model2, /*strict=*/false));
}
