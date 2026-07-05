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
