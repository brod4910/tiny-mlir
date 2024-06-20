#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/Support/LogicalResult.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyOps.cpp.inc"

namespace mlir::tiny {

void TinyDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tiny/Dialect/Tiny/IR/TinyTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tiny/Dialect/Tiny/IR/TinyOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tiny/Dialect/Tiny/IR/TinyAttrs.cpp.inc"
      >();
}

Operation *TinyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return ConstantOp::materialize(builder, value, type, loc);
}
} // namespace mlir::tiny