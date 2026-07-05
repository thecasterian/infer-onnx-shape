# Design: infer-onnx-shape

## Overview

A C++ CLI tool that reads an ONNX model, runs the official ONNX
shape-inference pass to annotate every intermediate tensor with its inferred
type and shape, and writes the annotated model back out as a valid ONNX file.
It works even when the model's weights live in a separate file that is not
provided.

## Goals

- Read an ONNX model file.
- Infer the shape (and element type) of all tensors in the model graph.
- Write the inferred shapes into an output file that is itself a valid ONNX
  model (the input model annotated with `value_info`).
- Tolerate missing external weight data: never require the weight file to be
  present in order to run shape inference.

## Non-goals

- Reconstructing or synthesizing weight values.
- Resolving value-dependent shapes when the required data lives in an absent
  external weight file (see Known limitations).
- Overriding or supplying input shapes from the CLI (may be added later; not
  in scope now).

## Core approach

Use the official ONNX C++ library. Parsing is done directly against the
protobuf `ModelProto`; shape inference is delegated to
`onnx::shape_inference::InferShapes`, which populates the graph's `value_info`
with the inferred type/shape of every intermediate tensor. Serializing the
resulting `ModelProto` back out yields an ONNX file where all tensor shapes
are recorded.

## Architecture and data flow

```
input.onnx --> [Loader] --> ModelProto --> [Inference] --> annotated ModelProto --> [Writer] --> output.onnx
                  |                             |
        parse protobuf only            onnx::shape_inference::
        (no external weights)          InferShapes() fills value_info
```

Four small, independently-testable components:

1. **CLI** (`main.cpp`)
   - Usage: `infer-onnx-shape <input.onnx> <output.onnx> [--strict]`
   - Validates arguments, dispatches to the components, sets the process exit
     code, prints a short summary.

2. **Loader** (`loader.{h,cpp}`)
   - `onnx::ModelProto load(const std::string& path)`
   - Parses the protobuf directly (`ParseFromIstream`). Deliberately does NOT
     invoke ONNX's external-data loader.
   - Rationale: initializer dims and element types live in the `TensorProto`
     inside the model file itself; only the raw bytes are external. Shape
     inference does not need the bytes, so a missing weight file does not
     block the tool.

3. **Inference** (`infer.{h,cpp}`)
   - Wraps `onnx::shape_inference::InferShapes(model, ...)`.
   - Default mode is lenient: nodes whose shape cannot be inferred are skipped
     rather than fatal, maximizing annotation coverage.
   - `--strict` opts into failing on the first inference error.

4. **Writer** (`writer.{h,cpp}`)
   - `void save(const onnx::ModelProto& model, const std::string& path)` via
     `SerializeToOstream`.
   - Preserves the full model as-is (graph, nodes, initializer metadata) and
     adds the newly inferred `value_info`.

## Output content

The output ONNX mirrors the input model in full (graph structure, nodes,
initializers / weight metadata) with the inferred `value_info` added. It is a
drop-in, shape-annotated replacement usable by downstream tools.

## Error handling

- Missing or unreadable input file, or protobuf parse failure: message to
  stderr, non-zero exit code.
- Inference failures: lenient by default (log a warning, continue). `--strict`
  makes them fatal.
- Print a summary: number of tensors annotated, number left unresolved.

## Known limitations

Value-dependent operators (for example `Reshape` or `Slice` whose shape
operand is an externally-stored initializer) may leave some downstream shapes
unresolved when the weight file is absent, because those bytes are genuinely
unavailable. Everything structurally inferable is still annotated.

## Build

- CMake with `FetchContent` pulling `onnx` (which brings protobuf in).
- C++17.
- Binary name: `infer-onnx-shape`.

## Testing

- Framework: GoogleTest (via FetchContent).
- Test 1: construct a tiny `ModelProto` in memory (e.g. `Conv -> Relu`, or
  `Gemm`) with declared input shapes but no weight data; run the inference
  component; assert `value_info` is populated with the expected shapes.
- Test 2: build a model with an external initializer reference and no backing
  data file; assert the tool still loads and annotates it.
- Wire tests into CTest.
