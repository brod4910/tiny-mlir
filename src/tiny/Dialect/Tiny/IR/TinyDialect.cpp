#include "TinyDialect.h"
#include "TinyOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "tiny/Dialect/Tiny/IR/TinyDialect.cpp.inc"
#define GET_OP_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyOps.cpp.inc"

namespace mlir::tiny {

void TinyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiny/Dialect/Tiny/IR/TinyOps.cpp.inc"
      >();
}

} // namespace mlir::tiny