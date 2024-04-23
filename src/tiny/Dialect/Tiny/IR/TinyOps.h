#pragma once

#include "TinyDialect.h"
#include "mlir/IR/BuiltinOps.h"                   // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                 // from @llvm-project
#include "mlir/IR/Dialect.h"                      // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project

#define GET_OP_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyOps.h.inc"