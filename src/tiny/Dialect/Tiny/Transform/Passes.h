#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::tiny {
#define GEN_PASS_DECL
#include "tiny/Dialect/Tiny/Transform/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//
#define GEN_PASS_REGISTRATION
#include "tiny/Dialect/Tiny/Transform/Passes.h.inc"
} // namespace mlir::tiny