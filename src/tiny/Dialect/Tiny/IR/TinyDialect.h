#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include "mlir/Support/LogicalResult.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h.inc"

#include <iostream>

namespace mlir::tiny {
/*
---------------------------------------------------
---------------------- Traits ---------------------
--------------------------------------------------- */

/* -------- ElementwiseBroadcastable Trait -------- */

bool verifyElementwise(Operation *op);
bool hasElementwiseBroadcastableTrait(Operation *op);
bool isElementwiseBroadcastableOpOnRankedTensors(Operation *op);

bool isElementwiseOpOnRankedTensors(Operation *op);

template <typename ConcreteType>
struct ElementwiseBroadcastable
    : public OpTrait::TraitBase<ConcreteType, ElementwiseBroadcastable> {
  static LogicalResult verifyTrait(Operation *op) {
    auto isMappableType = [](Type type) {
      return llvm::isa<RankedTensorType>(type);
    };

    auto resultMappableTypes = llvm::to_vector(
        llvm::make_filter_range(op->getResultTypes(), isMappableType));
    auto operandsMappableTypes = llvm::to_vector(
        llvm::make_filter_range(op->getOperandTypes(), isMappableType));

    if (resultMappableTypes.empty() && operandsMappableTypes.empty()) {
      return success();
    }

    // TODO: Refactor opportunity here, inline verify elementwise here.
    auto isElementwise = verifyElementwise(op);

    SmallVector<SmallVector<int64_t, 6>> shapes;

    for (auto type : op->getOperandTypes()) {
      auto shapedType = cast<ShapedType>(type);
      shapes.emplace_back(shapedType.getShape());
    }

    auto isBroadcastable = OpTrait::util::staticallyKnownBroadcastable(shapes);

    if (!isElementwise && !isBroadcastable) {
      return op->emitOpError(
          "operands must be elementwise mappable or broadcastable.");
    }

    return success();
  }
};

/* ---------------- Reducer Trait ----------------- */

bool hasReducerTrait(Operation *op);

bool isReducerOpOnRankedTensors(Operation *op);

template <typename ConcreteType>
struct Reducer : public OpTrait::TraitBase<ConcreteType, Reducer> {
  static LogicalResult verifyTrait(Operation *op) {
    auto isMappableType = [](Type type) {
      return llvm::isa<RankedTensorType>(type);
    };

    if (!isMappableType(op->getOperand(0).getType()) &&
        !isMappableType(op->getResult(0).getType())) {
      return success();
    }

    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return op->emitOpError("reducer requires 1 operand and 1 result.");
    }

    return success();
  }
};

} // namespace mlir::tiny

#define GET_TYPEDEF_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyTypes.h.inc"

#include "tiny/Dialect/Tiny/IR/TinyAttrsEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyAttrs.h.inc"

#define GET_OP_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyOps.h.inc"
