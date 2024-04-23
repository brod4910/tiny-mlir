#include "tiny/Dialect/Tiny/TinyDialect.h"
#include "mlir/IR/Builders.h"
#include "tiny/Dialect/Tiny/TinyOps.h"
#include "llvm/ADT/TypeSwitch.h"

#include "tiny/Dialect/Tiny/TinyDialect.cpp.inc"
#define GET_OP_CLASSES
#include "tiny/Dialect/Tiny/TinyOps.cpp.inc"

namespace mlir::tiny {

void TinyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiny/Dialect/Tiny/TinyOps.cpp.inc"
      >();
}

} // namespace mlir::tiny