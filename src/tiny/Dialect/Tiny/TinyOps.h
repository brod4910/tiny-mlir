#pragma once

#include "mlir/IR/BuiltinOps.h"                   // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                 // from @llvm-project
#include "mlir/IR/Dialect.h"                      // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "src/tiny/Dialect/Tiny/TinyDialect.h"

#define GET_OP_CLASSES
#include "src/tiny/Dialect/Tiny/TinyOps.h.inc"