#include "tiny/Dialect/Accelerator/Transform//AcclConversion.h"
#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

namespace mlir {
AcclTypeConverter::AcclTypeConverter(MLIRContext *context, int numWarps,
                                     int threadsPerWarp)
    : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp) {
  // This is required?
  addConversion([](Type type) { return type; });

  addConversion([this](RankedTensorType type) -> RankedTensorType {
    if (type.getEncoding()) {
      return type;
    }
    tiny::accl::CTALayoutAttr encoding =
        tiny::accl::getDefaultCTALayout(this->context, type.getShape());
    return RankedTensorType::get(type.getShape(), type.getElementType(),
                                 encoding);
  });
}

AcclConversionTarget::AcclConversionTarget(MLIRContext &ctx,
                                           AcclTypeConverter &typeConverter)
    : ConversionTarget(ctx) {
  addLegalDialect<tiny::accl::AcclDialect>();
  addDynamicallyLegalDialect<tiny::TinyDialect, scf::SCFDialect>(
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