#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include <sys/_types/_int64_t.h>

#include "mlir/Support/LogicalResult.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.cpp.inc"

#define GET_OP_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyAttrs.cpp.inc"

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

// Copied from ACCL dialect.
static LogicalResult parseIntAttrValue(AsmParser &parser,
                                       const NamedAttribute &attr, int &value,
                                       StringRef desc) {
  auto intAttr = dyn_cast<IntegerAttr>(attr.getValue());
  if (!intAttr) {
    parser.emitError(parser.getNameLoc(), "expected an integer type in ")
        << desc;
    return failure();
  }
  if (intAttr.getType().isSignedInteger()) {
    int64_t attrVal = intAttr.getSInt();
    value = attrVal;
  } else if (intAttr.getType().isSignlessInteger()) {
    int64_t attrVal = intAttr.getInt();
    value = attrVal;
  } else {
    value = intAttr.getUInt();
  }
  return success();
}
} // namespace mlir::tiny