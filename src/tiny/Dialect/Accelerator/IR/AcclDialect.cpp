#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "tiny/Dialect/Accelerator/IR/AcclDialect.cpp.inc"

namespace mlir::tiny::accl {
void AcclDialect::initialize() {}

Operation *AcclDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return tiny::ConstantOp::materialize(builder, value, type, loc);
}
} // namespace mlir::tiny::accl