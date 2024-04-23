#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::tiny {
#define GEN_PASS_DECL
#include "tiny/Transform/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "tiny/Transform/Passes.h.inc"
} // namespace mlir::tiny

namespace mlir::tiny {} // namespace mlir::tiny