#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "app.h"
#include "model_builders.h"

namespace fs = std::filesystem;

TEST(App, EndToEndAnnotatesOutputModel) {
  fs::path dir = fs::temp_directory_path() / "ios_app";
  fs::remove_all(dir);
  fs::create_directories(dir);

  onnx::GraphProto g;
  ios::test::add_input(&g, "X", onnx::TensorProto::FLOAT, {2, 3});
  ios::test::add_node(&g, "Relu", {"X"}, {"H"});
  ios::test::add_node(&g, "Relu", {"H"}, {"Y"});
  ios::test::add_output(&g, "Y");
  auto model = ios::test::make_model(g);

  auto in_path = (dir / "in.onnx").string();
  auto out_path = (dir / "out.onnx").string();
  {
    std::ofstream f(in_path, std::ios::binary);
    model.SerializeToOstream(&f);
  }

  int rc = ios::run(in_path, out_path, /*strict=*/false);
  EXPECT_EQ(rc, 0);

  std::ifstream out(out_path, std::ios::binary);
  onnx::ModelProto result;
  ASSERT_TRUE(result.ParseFromIstream(&out));
  auto h = ios::test::get_shape(result.graph(), "H");
  ASSERT_TRUE(h.has_value());
  EXPECT_EQ(*h, (std::vector<int64_t>{2, 3}));
}

TEST(App, ReturnsNonZeroOnBadInput) {
  int rc = ios::run("/no/such/input.onnx",
                    (fs::temp_directory_path() / "unused.onnx").string(),
                    /*strict=*/false);
  EXPECT_NE(rc, 0);
}
