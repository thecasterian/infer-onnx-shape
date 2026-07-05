#include <gtest/gtest.h>

#include "model_builders.h"

// Verifies the ONNX/protobuf/googletest toolchain compiles and links.
TEST(Smoke, BuildsAModelProto) {
  onnx::GraphProto g;
  ios::test::add_input(&g, "X", onnx::TensorProto::FLOAT, {2, 3});
  ios::test::add_node(&g, "Relu", {"X"}, {"Y"});
  ios::test::add_output(&g, "Y");
  auto model = ios::test::make_model(g);

  EXPECT_EQ(model.graph().node_size(), 1);
  EXPECT_EQ(model.graph().node(0).op_type(), "Relu");
}
