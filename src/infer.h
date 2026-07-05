#pragma once

#include <onnx/onnx_pb.h>

namespace ios {

struct InferResult {
  int annotated = 0;          // intermediate tensors that got a shape
  int total_intermediate = 0; // intermediate tensors (node outputs, non-graph-io)
};

// Run ONNX shape inference (data propagation ON) in-place on `model`,
// populating graph.value_info. When `strict` is true, the first inference
// error throws (onnx::InferenceError, a std::runtime_error); otherwise
// failing nodes are skipped.
InferResult infer_shapes(onnx::ModelProto& model, bool strict);

}  // namespace ios
