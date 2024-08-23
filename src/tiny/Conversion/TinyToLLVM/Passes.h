#pragma once

#include "tiny/Conversion/TinyToLLVM/TinyToLLVMPass.h"

namespace mlir::tiny {
#define GEN_PASS_REGISTRATION
#include "tiny/Conversion/TinyToLLVM/Passes.h.inc"
} // namespace mlir::tiny