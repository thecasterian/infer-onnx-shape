#include "infer.h"

#include <onnx/defs/schema.h>
#include <onnx/shape_inference/implementation.h>

#include <set>
#include <string>

namespace ios {

InferResult infer_shapes(onnx::ModelProto& model, bool strict) {
  onnx::ShapeInferenceOptions options(
      /*check_type=*/false,
      /*error_mode=*/strict ? 1 : 0,
      /*enable_data_propagation=*/true);
  onnx::shape_inference::InferShapes(
      model, onnx::OpSchemaRegistry::Instance(), options);

  const auto& g = model.graph();

  std::set<std::string> declared;
  for (const auto& i : g.input()) declared.insert(i.name());
  for (const auto& o : g.output()) declared.insert(o.name());
  for (const auto& t : g.initializer()) declared.insert(t.name());

  std::set<std::string> intermediate;
  for (const auto& n : g.node())
    for (const auto& out : n.output())
      if (!out.empty() && declared.find(out) == declared.end())
        intermediate.insert(out);

  int annotated = 0;
  for (const auto& vi : g.value_info())
    if (intermediate.find(vi.name()) != intermediate.end() &&
        vi.type().tensor_type().has_shape())
      ++annotated;

  InferResult r;
  r.total_intermediate = static_cast<int>(intermediate.size());
  r.annotated = annotated;
  return r;
}

}  // namespace ios
