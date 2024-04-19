#include "src/tiny/Dialect/Tiny/TinyDialect.h"
#include "mlir/IR/Builders.h"
#include "src/tiny/Dialect/Tiny/TinyOps.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/tiny/Dialect/Tiny/TinyDialect.cpp.inc"
#define GET_OP_CLASSES
#include "src/tiny/Dialect/Tiny/TinyOps.cpp.inc"

namespace mlir::tiny {

void TinyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/tiny/Dialect/Tiny/TinyOps.cpp.inc"
      >();
}

} // namespace mlir::tiny