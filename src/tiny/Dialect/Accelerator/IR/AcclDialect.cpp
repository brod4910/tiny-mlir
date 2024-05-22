#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#define GET_ATTRDEF_CLASSES
#include "tiny/Dialect/Accelerator/IR/AcclAttrs.cpp.inc"

#include "tiny/Dialect/Accelerator/IR/AcclDialect.cpp.inc"
#include <algorithm>
#include <cstdint>
#include <numeric>

namespace mlir::tiny::accl {
void AcclDialect::initialize() {}

Operation *AcclDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return tiny::ConstantOp::materialize(builder, value, type, loc);
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

CTALayoutAttr getDefaultCTALayout(MLIRContext *context,
                                  ArrayRef<int64_t> shape) {
  int rank = shape.size();
  constexpr uint64_t threadsPerWarp = 256;
  uint64_t numel = std::accumulate(shape.begin(), shape.end(), 0);

  uint64_t numWarps = ceil_div(numel, threadsPerWarp);
  auto threadBlockTile = ThreadBlockTileAttr::get(context, 1, 1, 1);
  auto warpTile = WarpTileAttr::get(context, numWarps, 1, 1);
  auto threadTile = ThreadTileAttr::get(context, threadsPerWarp, 1, 1);
  return CTALayoutAttr::get(context, threadBlockTile, warpTile, threadTile);
}

Attribute getDefaultMMAEncoding(MLIRContext *context, int numWarps,
                                int threadsPerWarp) {
  return nullptr;
}

} // namespace mlir::tiny::accl