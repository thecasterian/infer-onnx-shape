#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "infer.h"
#include "loader.h"
#include "model_builders.h"

namespace fs = std::filesystem;

namespace {

// Build a Reshape model whose shape operand "newshape" is an EXTERNAL int64
// initializer of two values, and serialize it to <dir>/model.onnx.
// Does NOT write the backing data file.
void write_external_reshape_model(const fs::path& dir,
                                  const std::string& data_location,
                                  const std::vector<int64_t>& shape_vals) {
  onnx::GraphProto g;
  ios::test::add_input(&g, "X", onnx::TensorProto::FLOAT, {2, 3, 4});
  auto* t = g.add_initializer();
  t->set_name("newshape");
  t->set_data_type(onnx::TensorProto::INT64);
  t->add_dims(static_cast<int64_t>(shape_vals.size()));
  ios::test::set_external(
      t, data_location, /*offset=*/0,
      /*length=*/static_cast<int64_t>(shape_vals.size() * sizeof(int64_t)));
  ios::test::add_node(&g, "Reshape", {"X", "newshape"}, {"H"});
  ios::test::add_node(&g, "Relu", {"H"}, {"Y"});
  ios::test::add_output(&g, "Y");
  auto model = ios::test::make_model(g);

  std::ofstream f(dir / "model.onnx", std::ios::binary);
  model.SerializeToOstream(&f);
}

}  // namespace

TEST(Loader, ThrowsOnMissingInput) {
  EXPECT_THROW(ios::load("/no/such/model.onnx"), std::runtime_error);
}

TEST(Loader, LoadsExternalInitializerAndResolvesShape) {
  fs::path dir = fs::temp_directory_path() / "ios_ext_present";
  fs::remove_all(dir);
  fs::create_directories(dir);

  std::vector<int64_t> vals{6, 4};
  {
    std::ofstream f(dir / "weights.bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(vals.data()),
            vals.size() * sizeof(int64_t));
  }
  write_external_reshape_model(dir, "weights.bin", vals);

  auto model = ios::load((dir / "model.onnx").string());

  const auto& init = model.graph().initializer(0);
  EXPECT_EQ(init.data_location(), onnx::TensorProto::DEFAULT);
  EXPECT_EQ(init.raw_data().size(), vals.size() * sizeof(int64_t));

  ios::infer_shapes(model, /*strict=*/false);
  auto h = ios::test::get_shape(model.graph(), "H");
  ASSERT_TRUE(h.has_value());
  EXPECT_EQ(*h, (std::vector<int64_t>{6, 4}));
}

TEST(Loader, ToleratesMissingExternalFile) {
  fs::path dir = fs::temp_directory_path() / "ios_ext_absent";
  fs::remove_all(dir);
  fs::create_directories(dir);

  // Note: weights.bin is intentionally NOT created.
  write_external_reshape_model(dir, "weights.bin", {6, 4});

  onnx::ModelProto model;
  ASSERT_NO_THROW(model = ios::load((dir / "model.onnx").string()));
  // Model still parsed fully; the initializer keeps its EXTERNAL marker.
  EXPECT_EQ(model.graph().node_size(), 2);
  EXPECT_EQ(model.graph().initializer(0).data_location(),
            onnx::TensorProto::EXTERNAL);
}
