# infer-onnx-shape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a C++ CLI that reads an ONNX model, infers the shape/type of every tensor (resolving value-dependent shapes via ONNX data propagation), and writes the annotated model back out as a valid ONNX file — tolerating a missing external weights file.

**Architecture:** A small static library (`ios_core`) split into four focused units — Loader (parse + external-data loading), Inference (`onnx::shape_inference::InferShapes` with data propagation), Writer (serialize), and App (orchestration) — plus a thin `main.cpp`. The official ONNX C++ library does the heavy lifting; we own I/O, external-data loading, and error handling.

**Tech Stack:** C++17, CMake with FetchContent (onnx, protobuf, googletest), ONNX C++ library, GoogleTest.

---

## Conventions

- **Namespace:** all our code lives in `namespace ios`; test helpers in `namespace ios::test`.
- **ONNX namespace:** `onnx::` (e.g. `onnx::ModelProto`, `onnx::TensorProto`).
- **Errors:** components throw `std::runtime_error`; only `App::run` catches and maps to an exit code.
- **Commits:** single-line, imperative, capitalized, no Conventional-Commits prefix.
- **Build/test commands** (run from repo root):
  - Configure: `cmake -S . -B build`
  - Build: `cmake --build build -j`
  - Test: `ctest --test-dir build --output-on-failure`

---

## Task 1: Project scaffold, dependencies, and smoke test

**Files:**
- Create: `CMakeLists.txt`
- Create: `tests/model_builders.h`
- Create: `tests/smoke_test.cpp`
- Create: `.gitignore`

- [ ] **Step 1: Write `.gitignore`**

Create `.gitignore`:

```gitignore
/build/
```

- [ ] **Step 2: Write `CMakeLists.txt`**

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.24)
project(infer_onnx_shape CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# ---- protobuf ----
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v3.21.12
)
FetchContent_MakeAvailable(protobuf)

# ---- onnx ----
set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(ONNX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(ONNX_USE_PROTOBUF_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ONNX_CUSTOM_PROTOC_EXECUTABLE "$<TARGET_FILE:protobuf::protoc>" CACHE STRING "" FORCE)
FetchContent_Declare(
  onnx
  GIT_REPOSITORY https://github.com/onnx/onnx.git
  GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(onnx)

# ---- core library ----
add_library(ios_core
  src/loader.cpp
  src/infer.cpp
  src/writer.cpp
  src/app.cpp
)
target_include_directories(ios_core PUBLIC src)
target_link_libraries(ios_core PUBLIC onnx)

# ---- CLI ----
add_executable(infer-onnx-shape src/main.cpp)
target_link_libraries(infer-onnx-shape PRIVATE ios_core)

# ---- tests ----
enable_testing()
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(ios_tests
  tests/smoke_test.cpp
)
target_link_libraries(ios_tests PRIVATE ios_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(ios_tests)
```

> **NOTE (build integration risk):** onnx generates its `.pb.cc` at build time using protoc. The `ONNX_CUSTOM_PROTOC_EXECUTABLE` genex points onnx at the FetchContent-built `protobuf::protoc`. If configure/build fails on protobuf resolution, fall back to **system protobuf**: on Arch run `sudo pacman -S --needed protobuf`, then in `CMakeLists.txt` delete the entire `# ---- protobuf ----` block and the `ONNX_CUSTOM_PROTOC_EXECUTABLE`/`ONNX_USE_PROTOBUF_SHARED_LIBS` lines (onnx will `find_package(Protobuf)` on its own). Keep onnx and googletest via FetchContent. The task is complete only when Step 6 builds and passes.

- [ ] **Step 3: Create `src/` placeholder sources so the library links**

The `ios_core` target lists four `.cpp` files that later tasks fill in. Create minimal compilable stubs now so Task 1 builds in isolation. Create `src/loader.cpp`, `src/infer.cpp`, `src/writer.cpp`, `src/app.cpp` each containing exactly:

```cpp
// filled in by a later task
```

(An empty translation unit is valid C++ and links fine.)

- [ ] **Step 4: Write `tests/model_builders.h`**

Create `tests/model_builders.h`:

```cpp
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
```

- [ ] **Step 5: Write the smoke test**

Create `tests/smoke_test.cpp`:

```cpp
#include <gtest/gtest.h>

#include "model_builders.h"

// Verifies the ONNX/protobuf/googletest toolchain compiles and links.
TEST(Smoke, BuildsAModelProto) {
  onnx::GraphProto g;
  ios::test::add_input(&g, "X", onnx::TensorProto::FLOAT, {2, 3});
  ios::test::add_node(&g, "Relu", {"X"}, {"Y"});
  ios::test::add_output(&g, "Y");
  auto model = ios::test::make_model(g);

  EXPECT_EQ(model.graph().node_size(), 1);
  EXPECT_EQ(model.graph().node(0).op_type(), "Relu");
}
```

- [ ] **Step 6: Configure, build, and run**

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: configure + build succeed (first build compiles protobuf + onnx, so it is slow — several minutes), and `Smoke.BuildsAModelProto` passes. If the build fails on protobuf, apply the fallback in the Step 2 NOTE and retry.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt .gitignore tests/model_builders.h tests/smoke_test.cpp src/loader.cpp src/infer.cpp src/writer.cpp src/app.cpp
git commit -m "Scaffold build with ONNX FetchContent and smoke test"
```

---

## Task 2: Writer

**Files:**
- Create: `src/writer.h`
- Modify: `src/writer.cpp` (replace stub)
- Create: `tests/writer_test.cpp`
- Modify: `CMakeLists.txt` (add `tests/writer_test.cpp` to `ios_tests`)

- [ ] **Step 1: Write the failing test**

Create `tests/writer_test.cpp`:

```cpp
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
```

Add `tests/writer_test.cpp` to the `ios_tests` target in `CMakeLists.txt`:

```cmake
add_executable(ios_tests
  tests/smoke_test.cpp
  tests/writer_test.cpp
)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: FAIL to compile — `writer.h` not found / `ios::save` undefined.

- [ ] **Step 3: Write `src/writer.h`**

Create `src/writer.h`:

```cpp
#pragma once

#include <onnx/onnx_pb.h>

#include <string>

namespace ios {

// Serialize `model` to `path` in ONNX (protobuf) binary format.
// Throws std::runtime_error if the file cannot be opened or written.
void save(const onnx::ModelProto& model, const std::string& path);

}  // namespace ios
```

- [ ] **Step 4: Write `src/writer.cpp`**

Replace the stub in `src/writer.cpp` with:

```cpp
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R Writer`
Expected: PASS (`Writer.SerializesAndRoundTrips`, `Writer.ThrowsOnUnwritablePath`).

- [ ] **Step 6: Commit**

```bash
git add src/writer.h src/writer.cpp tests/writer_test.cpp CMakeLists.txt
git commit -m "Add model writer with serialization round-trip test"
```

---

## Task 3: Shape inference with data propagation

**Files:**
- Create: `src/infer.h`
- Modify: `src/infer.cpp` (replace stub)
- Create: `tests/infer_test.cpp`
- Modify: `CMakeLists.txt` (add `tests/infer_test.cpp` to `ios_tests`)

- [ ] **Step 1: Write the failing test**

Create `tests/infer_test.cpp`:

```cpp
#include <gtest/gtest.h>

#include "infer.h"
#include "model_builders.h"

// Structural inference: an intermediate tensor gets a value_info shape.
TEST(Inference, ReluChainAnnotatesIntermediate) {
  onnx::GraphProto g;
  ios::test::add_input(&g, "X", onnx::TensorProto::FLOAT, {2, 3});
  ios::test::add_node(&g, "Relu", {"X"}, {"H"});
  ios::test::add_node(&g, "Relu", {"H"}, {"Y"});
  ios::test::add_output(&g, "Y");
  auto model = ios::test::make_model(g);

  auto r = ios::infer_shapes(model, /*strict=*/false);

  auto h = ios::test::get_shape(model.graph(), "H");
  ASSERT_TRUE(h.has_value());
  EXPECT_EQ(*h, (std::vector<int64_t>{2, 3}));
  EXPECT_GE(r.annotated, 1);
  EXPECT_EQ(r.total_intermediate, 1);
}

// Data propagation: Reshape output resolved from an inline shape initializer.
TEST(Inference, ReshapeResolvesFromInlineInitializer) {
  onnx::GraphProto g;
  ios::test::add_input(&g, "X", onnx::TensorProto::FLOAT, {2, 3, 4});
  ios::test::add_int64_initializer(&g, "newshape", {2}, {6, 4});
  ios::test::add_node(&g, "Reshape", {"X", "newshape"}, {"H"});
  ios::test::add_node(&g, "Relu", {"H"}, {"Y"});
  ios::test::add_output(&g, "Y");
  auto model = ios::test::make_model(g);

  ios::infer_shapes(model, /*strict=*/false);

  auto h = ios::test::get_shape(model.graph(), "H");
  ASSERT_TRUE(h.has_value());
  EXPECT_EQ(*h, (std::vector<int64_t>{6, 4}));
}
```

Add `tests/infer_test.cpp` to the `ios_tests` target in `CMakeLists.txt`:

```cmake
add_executable(ios_tests
  tests/smoke_test.cpp
  tests/writer_test.cpp
  tests/infer_test.cpp
)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: FAIL to compile — `infer.h` not found / `ios::infer_shapes` undefined.

- [ ] **Step 3: Write `src/infer.h`**

Create `src/infer.h`:

```cpp
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
```

- [ ] **Step 4: Write `src/infer.cpp`**

Replace the stub in `src/infer.cpp` with:

```cpp
#include "infer.h"

#include <onnx/defs/schema.h>
#include <onnx/shape_inference/implementation.h>

#include <set>
#include <string>

namespace ios {

InferResult infer_shapes(onnx::ModelProto& model, bool strict) {
  onnx::shape_inference::ShapeInferenceOptions options(
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R Inference`
Expected: PASS (`Inference.ReluChainAnnotatesIntermediate`, `Inference.ReshapeResolvesFromInlineInitializer`).

> If `ReshapeResolvesFromInlineInitializer` fails with `H` symbolic, confirm `enable_data_propagation` is the third `ShapeInferenceOptions` arg for the pinned onnx v1.17.0 (check `build/_deps/onnx-src/onnx/shape_inference/implementation.h`); adjust the constructor call to match if the field order differs.

- [ ] **Step 6: Commit**

```bash
git add src/infer.h src/infer.cpp tests/infer_test.cpp CMakeLists.txt
git commit -m "Add shape inference with data propagation"
```

---

## Task 4: Loader with external-data loading

**Files:**
- Create: `src/loader.h`
- Modify: `src/loader.cpp` (replace stub)
- Create: `tests/loader_test.cpp`
- Modify: `CMakeLists.txt` (add `tests/loader_test.cpp` to `ios_tests`)

- [ ] **Step 1: Write the failing test**

Create `tests/loader_test.cpp`:

```cpp
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
```

Add `tests/loader_test.cpp` to the `ios_tests` target in `CMakeLists.txt`:

```cmake
add_executable(ios_tests
  tests/smoke_test.cpp
  tests/writer_test.cpp
  tests/infer_test.cpp
  tests/loader_test.cpp
)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: FAIL to compile — `loader.h` not found / `ios::load` undefined.

- [ ] **Step 3: Write `src/loader.h`**

Create `src/loader.h`:

```cpp
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
```

- [ ] **Step 4: Write `src/loader.cpp`**

Replace the stub in `src/loader.cpp` with:

```cpp
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R Loader`
Expected: PASS (`Loader.ThrowsOnMissingInput`, `Loader.LoadsExternalInitializerAndResolvesShape`, `Loader.ToleratesMissingExternalFile`).

- [ ] **Step 6: Commit**

```bash
git add src/loader.h src/loader.cpp tests/loader_test.cpp CMakeLists.txt
git commit -m "Add loader with external-data loading and missing-file tolerance"
```

---

## Task 5: App orchestration and CLI

**Files:**
- Create: `src/app.h`
- Modify: `src/app.cpp` (replace stub)
- Create: `src/main.cpp`
- Create: `tests/app_test.cpp`
- Modify: `CMakeLists.txt` (add `tests/app_test.cpp` to `ios_tests`)

- [ ] **Step 1: Write the failing test**

Create `tests/app_test.cpp`:

```cpp
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
```

Add `tests/app_test.cpp` to the `ios_tests` target in `CMakeLists.txt`:

```cmake
add_executable(ios_tests
  tests/smoke_test.cpp
  tests/writer_test.cpp
  tests/infer_test.cpp
  tests/loader_test.cpp
  tests/app_test.cpp
)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: FAIL to compile — `app.h` not found / `ios::run` undefined.

- [ ] **Step 3: Write `src/app.h`**

Create `src/app.h`:

```cpp
#pragma once

#include <string>

namespace ios {

// Load `input`, infer shapes (data propagation on; `strict` = fail on first
// inference error), and write the annotated model to `output`. Prints a
// one-line summary on success. Returns 0 on success, 1 on error.
int run(const std::string& input, const std::string& output, bool strict);

}  // namespace ios
```

- [ ] **Step 4: Write `src/app.cpp`**

Replace the stub in `src/app.cpp` with:

```cpp
#include "app.h"

#include <exception>
#include <iostream>

#include "infer.h"
#include "loader.h"
#include "writer.h"

namespace ios {

int run(const std::string& input, const std::string& output, bool strict) {
  try {
    onnx::ModelProto model = load(input);
    InferResult r = infer_shapes(model, strict);
    save(model, output);
    std::cout << "Inferred shapes: " << r.annotated << "/"
              << r.total_intermediate << " intermediate tensors annotated. "
              << "Wrote " << output << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace ios
```

- [ ] **Step 5: Write `src/main.cpp`**

Create `src/main.cpp`:

```cpp
#include <iostream>
#include <string>
#include <vector>

#include "app.h"

namespace {

void print_usage() {
  std::cerr << "usage: infer-onnx-shape <input.onnx> <output.onnx> [--strict]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> positional;
  bool strict = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--strict") {
      strict = true;
    } else if (arg == "-h" || arg == "--help") {
      print_usage();
      return 0;
    } else {
      positional.push_back(arg);
    }
  }

  if (positional.size() != 2) {
    print_usage();
    return 2;
  }

  return ios::run(positional[0], positional[1], strict);
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R App`
Expected: PASS (`App.EndToEndAnnotatesOutputModel`, `App.ReturnsNonZeroOnBadInput`).

- [ ] **Step 7: Run the full suite**

Run: `ctest --test-dir build --output-on-failure`
Expected: all tests pass (Smoke, Writer, Inference, Loader, App).

- [ ] **Step 8: Commit**

```bash
git add src/app.h src/app.cpp src/main.cpp tests/app_test.cpp CMakeLists.txt
git commit -m "Add app orchestration and CLI entry point"
```

---

## Task 6: End-to-end CLI verification and README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Exercise the built binary on a real file**

Build a tiny model with the test scaffolding is already covered; here just confirm the CLI runs on a serialized model. Reuse the app test's input by generating one via a throwaway command is unnecessary — instead verify the binary's help and error paths:

```bash
./build/infer-onnx-shape --help
./build/infer-onnx-shape           # no args
echo "exit: $?"                     # expect 2
./build/infer-onnx-shape /no/such.onnx out.onnx
echo "exit: $?"                     # expect 1
```

Expected: help prints usage and exits 0; no-args prints usage and exits 2; bad input prints `error: cannot open input file: ...` and exits 1.

- [ ] **Step 2: Write `README.md`**

Create `README.md`:

```markdown
# infer-onnx-shape

A C++ CLI that reads an ONNX model, infers the shape and element type of every
tensor in the graph, and writes the annotated model back out as a valid ONNX
file. Value-dependent shapes (e.g. `Reshape`, `Slice`) are resolved via ONNX
data propagation. The tool works even when the model's weights live in a
separate external file that is not provided.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

The first build fetches and compiles protobuf, ONNX, and GoogleTest via
CMake FetchContent, so it takes a few minutes.

## Usage

```bash
infer-onnx-shape <input.onnx> <output.onnx> [--strict]
```

- `<output.onnx>` is the input model with inferred shapes recorded in
  `value_info`.
- `--strict` makes the first shape-inference error fatal (default: lenient,
  skipping tensors it cannot infer).

## Notes

- Weight values stored in an external file are loaded (relative to the input
  model's directory) so value-dependent shapes can be resolved. If that file
  is absent, the tool still runs and annotates every structurally inferable
  tensor.

## Tests

```bash
ctest --test-dir build --output-on-failure
```
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Add README and verify CLI behavior"
```

---

## Self-review notes

- **Spec coverage:** Loader (Task 4) covers parse + external-data loading + missing-file tolerance; Inference (Task 3) covers `InferShapes` + data propagation for value-dependent ops; Writer (Task 2) covers full-model serialization with `value_info`; App/CLI (Task 5) covers the `<input> <output> [--strict]` interface, lenient/strict modes, and the summary line. Tests map to spec's Test 1 (Relu/structural), Test 2 (inline data prop), Test 3 (external present), Test 4 (external absent).
- **Type consistency:** `ios::load` → `onnx::ModelProto`; `ios::infer_shapes(model, bool)` → `InferResult{annotated,total_intermediate}`; `ios::save(model, path)`; `ios::run(input, output, strict)`. Names are consistent across tasks.
- **Known risk:** protobuf/onnx FetchContent integration (Task 1 NOTE) — fallback to system protobuf documented.
```
