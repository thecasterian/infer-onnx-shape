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
