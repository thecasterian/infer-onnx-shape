#include "writer.h"

#include <fstream>
#include <stdexcept>

namespace ios {

void save(const onnx::ModelProto& model, const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("cannot open output file: " + path);
  if (!model.SerializeToOstream(&out))
    throw std::runtime_error("failed to serialize model to: " + path);
}

}  // namespace ios
