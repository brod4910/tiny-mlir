#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class AcclTypeConverter : public TypeConverter {
public:
  AcclTypeConverter(MLIRContext *context, int numWarps, int threadsPerWarp);

  int getNumWarps() const { return numWarps; }
  int getThreadsPerWarp() const { return threadsPerWarp; }
  MLIRContext *getContext() const { return context; };

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
};

class AcclConversionTarget : public ConversionTarget {
public:
  explicit AcclConversionTarget(MLIRContext &ctx,
                                AcclTypeConverter &typeConverter);
};
} // namespace mlir