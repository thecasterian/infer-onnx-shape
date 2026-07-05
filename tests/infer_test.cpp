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
