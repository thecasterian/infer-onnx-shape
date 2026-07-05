#pragma once

#include <onnx/onnx_pb.h>

#include <string>

namespace ios {

// Load an ONNX model from `path`. Parses the protobuf, then loads any
// EXTERNAL initializer data from files referenced relative to the model's
// directory (so value-dependent shape inference can see the values). A
// referenced external file that is missing is skipped with a warning rather
// than treated as fatal. Throws std::runtime_error if the model file itself
// cannot be opened or parsed.
onnx::ModelProto load(const std::string& path);

}  // namespace ios
