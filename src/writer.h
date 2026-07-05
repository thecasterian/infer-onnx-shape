#pragma once

#include <onnx/onnx_pb.h>

#include <string>

namespace ios {

// Serialize `model` to `path` in ONNX (protobuf) binary format.
// Throws std::runtime_error if the file cannot be opened or written.
void save(const onnx::ModelProto& model, const std::string& path);

}  // namespace ios
