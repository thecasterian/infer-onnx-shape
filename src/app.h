#pragma once

#include <string>

namespace ios {

// Load `input`, infer shapes (data propagation on; `strict` = fail on first
// inference error), and write the annotated model to `output`. Prints a
// one-line summary on success. Returns 0 on success, 1 on error.
int run(const std::string& input, const std::string& output, bool strict);

}  // namespace ios
