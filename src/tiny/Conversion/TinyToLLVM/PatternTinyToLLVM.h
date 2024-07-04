#pragma once
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tiny {
void populateElementwiseOpToLLVM(LLVMTypeConverter *typeConverter,
                                 RewritePatternSet &patterns);
}