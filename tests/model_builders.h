#pragma once

#include <onnx/onnx_pb.h>

#include <optional>
#include <string>
#include <vector>

namespace ios::test {

// Set a tensor type + static shape on a TypeProto.
inline void set_shape(onnx::TypeProto* type, int elem_type,
                      const std::vector<int64_t>& dims) {
  auto* tt = type->mutable_tensor_type();
  tt->set_elem_type(elem_type);
  auto* shape = tt->mutable_shape();
  shape->clear_dim();
  for (int64_t d : dims) shape->add_dim()->set_dim_value(d);
}

inline onnx::ValueInfoProto* add_input(onnx::GraphProto* g, const std::string& name,
                                       int elem_type, const std::vector<int64_t>& dims) {
  auto* vi = g->add_input();
  vi->set_name(name);
  set_shape(vi->mutable_type(), elem_type, dims);
  return vi;
}

// Output with no declared type; shape inference fills it in.
inline onnx::ValueInfoProto* add_output(onnx::GraphProto* g, const std::string& name) {
  auto* vi = g->add_output();
  vi->set_name(name);
  return vi;
}

inline onnx::NodeProto* add_node(onnx::GraphProto* g, const std::string& op_type,
                                 const std::vector<std::string>& inputs,
                                 const std::vector<std::string>& outputs) {
  auto* n = g->add_node();
  n->set_op_type(op_type);
  for (const auto& i : inputs) n->add_input(i);
  for (const auto& o : outputs) n->add_output(o);
  return n;
}

inline void add_int64_initializer(onnx::GraphProto* g, const std::string& name,
                                  const std::vector<int64_t>& dims,
                                  const std::vector<int64_t>& vals) {
  auto* t = g->add_initializer();
  t->set_name(name);
  t->set_data_type(onnx::TensorProto::INT64);
  for (int64_t d : dims) t->add_dims(d);
  for (int64_t v : vals) t->add_int64_data(v);
}

// Mark a TensorProto as external data (location relative to model dir).
inline void set_external(onnx::TensorProto* t, const std::string& location,
                         int64_t offset, int64_t length) {
  t->set_data_location(onnx::TensorProto::EXTERNAL);
  auto add = [&](const std::string& k, const std::string& v) {
    auto* e = t->add_external_data();
    e->set_key(k);
    e->set_value(v);
  };
  add("location", location);
  add("offset", std::to_string(offset));
  add("length", std::to_string(length));
}

inline onnx::ModelProto make_model(const onnx::GraphProto& graph, int opset = 17) {
  onnx::ModelProto m;
  m.set_ir_version(onnx::IR_VERSION);
  auto* op = m.add_opset_import();
  op->set_domain("");
  op->set_version(opset);
  *m.mutable_graph() = graph;
  m.mutable_graph()->set_name("test");
  return m;
}

// Look up an inferred shape by tensor name across value_info/output/input.
// Returns dims (-1 for a symbolic/unknown dim), or nullopt if no shape.
inline std::optional<std::vector<int64_t>> get_shape(const onnx::GraphProto& g,
                                                     const std::string& name) {
  auto extract =
      [](const onnx::TypeProto& type) -> std::optional<std::vector<int64_t>> {
    if (!type.has_tensor_type() || !type.tensor_type().has_shape())
      return std::nullopt;
    std::vector<int64_t> dims;
    for (const auto& d : type.tensor_type().shape().dim())
      dims.push_back(d.has_dim_value() ? d.dim_value() : -1);
    return dims;
  };
  for (const auto& vi : g.value_info())
    if (vi.name() == name) return extract(vi.type());
  for (const auto& vi : g.output())
    if (vi.name() == name) return extract(vi.type());
  for (const auto& vi : g.input())
    if (vi.name() == name) return extract(vi.type());
  return std::nullopt;
}

}  // namespace ios::test
