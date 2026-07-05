cmake_minimum_required(VERSION 3.24)
project(infer_onnx_shape CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# ---- protobuf ----
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# ONNX exports its ONNXTargets set (build-tree export + install(EXPORT)), which
# transitively references the protobuf target. That target must belong to an
# export set or generation fails, so we leave protobuf install/export rules on.
set(protobuf_INSTALL ON CACHE BOOL "" FORCE)
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
target_include_directories(ios_tests PRIVATE tests)
target_link_libraries(ios_tests PRIVATE ios_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(ios_tests)
