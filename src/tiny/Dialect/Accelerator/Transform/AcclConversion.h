#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class AcclTypeConverter : public TypeConverter {
public:
  AcclTypeConverter(MLIRContext *context, int numWarps, int threadsPerWarp);

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
};
} // namespace mlir