# infer-onnx-shape

A C++ CLI that reads an ONNX model, infers the shape and element type of the
tensors in the graph, and writes the annotated model back out as a valid ONNX
file. Value-dependent shapes (e.g. `Reshape`, `Slice`) are resolved via ONNX
data propagation. The tool works even when the model's weights live in a
separate external file that is not provided.

## Build

    cmake -S . -B build
    cmake --build build -j

The first build fetches and compiles protobuf, ONNX, and GoogleTest via
CMake FetchContent, so it takes a few minutes.

## Usage

    infer-onnx-shape <input.onnx> <output.onnx> [--strict]

- `<output.onnx>` is the input model with inferred shapes recorded in
  `value_info`.
- `--strict` makes the first shape-inference error fatal (default: lenient,
  skipping tensors it cannot infer).

Exit codes: `0` success, `1` runtime error (e.g. unreadable/invalid input),
`2` usage error.

## Notes

- Weight values stored in an external file are loaded (relative to the input
  model's directory) so value-dependent shapes can be resolved. If that file
  is absent, the tool still runs and annotates every structurally inferable
  tensor.

## Tests

    ctest --test-dir build --output-on-failure
