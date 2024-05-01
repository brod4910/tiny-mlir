#pragma once

#include "TinyDialect.h"
#include "mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/IR/BuiltinTypes.h" // from @llvm-project
#include "mlir/IR/Dialect.h"      // from @llvm-project
#include <cstdint>

#define GET_TYPEDEF_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyTypes.h.inc"

#define GET_OP_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyOps.h.inc"

namespace mlir::tiny {
class ConstantOpInt : public tiny::ConstantOp {
public:
  using tiny::ConstantOp::ConstantOp;

  static void build(OpBuilder &builder, OperationState &result, int64_t value,
                    unsigned width);

  static void build(OpBuilder &builder, OperationState &result, int64_t value,
                    Type type);

  inline int64_t value() {
    return cast<IntegerAttr>(tiny::ConstantOp::getValue()).getInt();
  }

  static bool classof(Operation *op);
};

class ConstantOpFloat : public tiny::ConstantOp {
public:
  using tiny::ConstantOp::ConstantOp;

  static void build(OpBuilder &builder, OperationState &result, APFloat value,
                    Type type);

  inline APFloat value() {
    return cast<FloatAttr>(tiny::ConstantOp::getValue()).getValue();
  }

  static bool classof(Operation *op);
};
} // namespace mlir::tiny