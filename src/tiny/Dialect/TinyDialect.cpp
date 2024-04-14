#include "src/tiny/Dialect/TinyDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/tiny/Dialect/TinyDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "src/tiny/Dialect/TinyTypes.cpp.inc"
#define GET_OP_CLASSES
#include "src/tiny/Dialect/TinyOps.cpp.inc"

namespace mlir::tiny {

void TinyDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/tiny/Dialect/TinyTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/tiny/Dialect/TinyOps.cpp.inc"
      >();
}

void TinyDialect::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns) const {}

} // namespace mlir::tiny