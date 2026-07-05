#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "model_builders.h"
#include "writer.h"

TEST(Writer, SerializesAndRoundTrips) {
  onnx::GraphProto g;
  ios::test::add_input(&g, "X", onnx::TensorProto::FLOAT, {2, 3});
  ios::test::add_node(&g, "Relu", {"X"}, {"Y"});
  ios::test::add_output(&g, "Y");
  auto model = ios::test::make_model(g);

  auto path = (std::filesystem::temp_directory_path() / "ios_writer.onnx").string();
  ios::save(model, path);

  std::ifstream in(path, std::ios::binary);
  onnx::ModelProto reloaded;
  ASSERT_TRUE(reloaded.ParseFromIstream(&in));
  EXPECT_EQ(reloaded.graph().node_size(), 1);
  EXPECT_EQ(reloaded.graph().node(0).op_type(), "Relu");
}

TEST(Writer, ThrowsOnUnwritablePath) {
  onnx::ModelProto model;
  EXPECT_THROW(ios::save(model, "/nonexistent-dir/x/y/z.onnx"), std::runtime_error);
}
