#pragma once

#include "tiny/Conversion/TinyToAccl/TinyToAcclPass.h"

namespace mlir::tiny {
#define GEN_PASS_REGISTRATION
#include "tiny/Conversion/TinyToAccl/Passes.h.inc"
} // namespace mlir::tiny