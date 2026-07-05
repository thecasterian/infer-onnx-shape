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
- Overriding or supplying input shapes from the CLI (may be added later; not
  in scope now).

## Core approach

Use the official ONNX C++ library. Parsing is done directly against the
protobuf `ModelProto`; shape inference is delegated to
`onnx::shape_inference::InferShapes`, which populates the graph's `value_info`
with the inferred type/shape of every intermediate tensor. Serializing the
resulting `ModelProto` back out yields an ONNX file where all tensor shapes
are recorded.

Shape inference is run with data propagation enabled
(`ShapeInferenceOptions.enable_data_propagation = true`). This drives the
per-operator `PartialDataPropagationFunction` (e.g. `Shape`, `Reshape`,
`Slice`, `Concat`) so that value-dependent output shapes are computed from the
actual constant values carried by initializers, rather than left symbolic.
Inline initializer values are read directly by the library; values stored in
an external weight file are loaded by the Loader (below) before inference so
they are equally available.

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
   - Parses the protobuf directly (`ParseFromIstream`).
   - Then loads external initializer data: for each initializer with
     `data_location() == TensorProto::EXTERNAL`, reads the `external_data`
     entries (`location`, `offset`, `length`), opens the referenced file
     relative to the model's directory, reads the bytes into `raw_data`, and
     clears the external-data marker. This is the C++ equivalent of ONNX's
     `load_external_data_for_model`, so value-dependent shape inference can
     see the values.
   - Missing-weights tolerance: if a referenced external file does not exist,
     that initializer is left without data (a warning is logged) and loading
     continues. Structural inference still works because initializer dims and
     element types live in the `TensorProto` itself; only value-dependent
     shapes that specifically needed the absent bytes remain unresolved.

3. **Inference** (`infer.{h,cpp}`)
   - Wraps `onnx::shape_inference::InferShapes(model, registry, options)` with
     `ShapeInferenceOptions.enable_data_propagation = true` so value-dependent
     ops (`Reshape`, `Slice`, `Shape`, `Concat`, ...) resolve concrete output
     shapes from constant values.
   - Default mode is lenient: nodes whose shape cannot be inferred are skipped
     rather than fatal, maximizing annotation coverage.
   - `--strict` opts into failing on the first inference error (maps to
     `ShapeInferenceOptions.error_mode`).

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

Value-dependent shapes are resolved whenever the driving values are available
— that is, from inline initializers or from external initializers whose
backing file is present. They remain unresolved only when the needed values
come from a runtime graph input, or from an external file that is genuinely
absent. Everything structurally inferable is always annotated.

## Build

- CMake with `FetchContent` pulling `onnx` (which brings protobuf in).
- C++17.
- Binary name: `infer-onnx-shape`.

## Testing

- Framework: GoogleTest (via FetchContent).
- Test 1: construct a tiny `ModelProto` in memory (e.g. `Conv -> Relu`, or
  `Gemm`) with declared input shapes but no weight data; run the inference
  component; assert `value_info` is populated with the expected shapes.
- Test 2 (data propagation, inline): a `Reshape` whose shape operand is an
  inline `int64` initializer; assert the output tensor's shape is resolved to
  the concrete target shape (not left symbolic).
- Test 3 (external data present): a model with an external initializer plus a
  backing data file on disk; assert the Loader reads the bytes and the
  dependent shape is resolved.
- Test 4 (external data absent): the same model with the backing file
  removed; assert the tool still loads, logs a warning, and annotates every
  structurally inferable tensor.
- Wire tests into CTest.
