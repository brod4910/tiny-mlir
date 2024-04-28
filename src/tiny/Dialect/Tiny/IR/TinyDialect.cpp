#include "TinyDialect.h"
#include "TinyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/IR/Types.h"
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

Operation *TinyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  auto tensor_type = dyn_cast<TensorType>(type);
  auto const_value = dyn_cast<DenseElementsAttr>(value);

  if (!tensor_type || !const_value) {
    return nullptr;
  }

  return builder.create<ConstantOp>(loc, type, const_value);
}
} // namespace mlir::tiny