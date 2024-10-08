#include "tiny/Dialect/Accelerator/Transform/AcclConversion.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

namespace mlir {
AcclTypeConverter::AcclTypeConverter(MLIRContext *context, int numWarps,
                                     int threadsPerWarp)
    : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp) {
  // This is required?
  addConversion([](Type type) { return type; });
}

AcclConversionTarget::AcclConversionTarget(MLIRContext &ctx,
                                           AcclTypeConverter &typeConverter)
    : ConversionTarget(ctx) {
  addLegalDialect<tiny::accl::AcclDialect, func::FuncDialect>();
  addDynamicallyLegalDialect<scf::SCFDialect, cf::ControlFlowDialect>(
      [&](Operation *op) -> bool {
        bool hasLegalRegions = true;
        for (auto &region : op->getRegions()) {
          hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
        }
        if (hasLegalRegions && typeConverter.isLegal(op)) {
          return true;
        }
        return false;
      });
}
} // namespace mlir