#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#include "tiny/Dialect/Tiny/IR/TinyDialect.h.inc"

namespace mlir::tiny {
/*
---------------------------------------------------
---------------------- Traits ---------------------
--------------------------------------------------- */
bool verifyElementwise(Operation *op);
bool hasElementwiseBroadcastableTrait(Operation *op);
bool isElementwiseBroadcastableOpOnRankedTensors(Operation *op);

template <typename ConcreteType>
struct ElementwiseBroadcastable
    : public OpTrait::TraitBase<ConcreteType, ElementwiseBroadcastable> {
  static LogicalResult verifyTrait(Operation *op) {
    auto isElementwise = verifyElementwise(op);

    auto operandTypes = op->getOperandTypes();
    SmallVector<SmallVector<int64_t, 6>> shapes;

    for (auto type : operandTypes) {
      auto tensorType = dyn_cast<RankedTensorType>(type);
      shapes.emplace_back(tensorType.getShape());
    }

    auto isBroadcastable = OpTrait::util::staticallyKnownBroadcastable(shapes);

    if (!isElementwise && !isBroadcastable) {
      return op->emitOpError(
          "operands must be elementwise mappable and broadcastable.");
    }

    return success();
  }
};
} // namespace mlir::tiny

#define GET_TYPEDEF_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyAttrs.h.inc"

#define GET_OP_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyOps.h.inc"
