#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "tiny/Dialect/Accelerator/IR/AcclAttrs.cpp.inc"
#include "tiny/Dialect/Accelerator/IR/AcclDialect.cpp.inc"

namespace mlir::tiny::accl {
void AcclDialect::initialize() {}

Operation *AcclDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return tiny::ConstantOp::materialize(builder, value, type, loc);
}

CTALayoutAttr getDefaultCTALayout(MLIRContext *context, ArrayRef<int64_t> shape,
                                  int numWarps, int threadsPerWarp) {
  int rank = shape.size();
  auto thread_block_tile = ThreadBlockTileAttr::get(context, 1, 1, 1);
  auto warp_tile = WarpTileAttr::get(context, 1, 1, 1);
  auto thread_tile = ThreadTileAttr::get(context, 1, 1, 1);
  return CTALayoutAttr::get(context, thread_block_tile, warp_tile, thread_tile)
}

Attribute getDefaultMMAEncoding(MLIRContext *context, int numWarps,
                                int threadsPerWarp) {
  return nullptr;
}

} // namespace mlir::tiny::accl