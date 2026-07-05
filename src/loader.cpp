#include "loader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace ios {
namespace {

void load_external_initializer(onnx::TensorProto& t, const fs::path& model_dir) {
  std::string location;
  int64_t offset = 0;
  int64_t length = -1;
  for (const auto& e : t.external_data()) {
    if (e.key() == "location")
      location = e.value();
    else if (e.key() == "offset")
      offset = std::stoll(e.value());
    else if (e.key() == "length")
      length = std::stoll(e.value());
  }
  if (location.empty()) return;

  fs::path full = model_dir.empty() ? fs::path(location) : model_dir / location;
  std::ifstream f(full, std::ios::binary);
  if (!f) {
    std::cerr << "warning: external data file not found; leaving initializer '"
              << t.name() << "' without values: " << full.string() << "\n";
    return;
  }
  f.seekg(offset, std::ios::beg);

  std::string buf;
  if (length < 0) {
    buf.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  } else {
    buf.resize(static_cast<size_t>(length));
    f.read(buf.data(), length);
    if (f.gcount() != length) {
      std::cerr << "warning: short read for external data of '" << t.name()
                << "'; leaving it unloaded\n";
      return;
    }
  }

  t.set_raw_data(buf);
  t.clear_data_location();
  t.clear_external_data();
}

}  // namespace

onnx::ModelProto load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open input file: " + path);

  onnx::ModelProto model;
  if (!model.ParseFromIstream(&in))
    throw std::runtime_error("failed to parse ONNX model: " + path);

  const fs::path model_dir = fs::path(path).parent_path();
  for (auto& t : *model.mutable_graph()->mutable_initializer())
    if (t.data_location() == onnx::TensorProto::EXTERNAL)
      load_external_initializer(t, model_dir);

  return model;
}

}  // namespace ios
