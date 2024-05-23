#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"
#include <functional>

#define GET_ATTRDEF_CLASSES
#include "tiny/Dialect/Accelerator/IR/AcclAttrs.cpp.inc"

#include "tiny/Dialect/Accelerator/IR/AcclDialect.cpp.inc"

#include <cstdint>
#include <iostream>
#include <numeric>

namespace mlir::tiny::accl {
void AcclDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tiny/Dialect/Accelerator/IR/AcclAttrs.cpp.inc"
      >();
}

Operation *AcclDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return tiny::ConstantOp::materialize(builder, value, type, loc);
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

CTALayoutAttr getDefaultCTALayout(MLIRContext *context,
                                  ArrayRef<int64_t> shape) {
  int rank = shape.size();
  constexpr uint64_t threadsPerWarp = 8;
  uint64_t numel =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
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

void CTALayoutAttr::print(mlir::AsmPrinter &printer) const {
  auto threadBlockTile = getThreadBlockTile();
  auto warpTile = getWarpTile();
  auto threadTile = getThreadTile();

  printer << "<{"
          << "threadBlockTile = [" << threadBlockTile.getM() << ", "
          << threadBlockTile.getN() << ", " << threadBlockTile.getK() << "]"
          << ", warpTile = [" << warpTile.getM() << ", " << warpTile.getN()
          << ", " << warpTile.getK() << "]"
          << ", threadTile = [" << threadTile.getM() << ", "
          << threadTile.getN() << ", " << threadTile.getK() << "]";

  printer << "}>";
}

} // namespace mlir::tiny::accl