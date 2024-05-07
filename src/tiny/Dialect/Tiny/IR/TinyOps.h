#pragma once

#include "TinyDialect.h"

#include <cstdint>

#include "mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/IR/BuiltinTypes.h" // from @llvm-project
#include "mlir/IR/Dialect.h"      // from @llvm-project

#define GET_TYPEDEF_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyTypes.h.inc"