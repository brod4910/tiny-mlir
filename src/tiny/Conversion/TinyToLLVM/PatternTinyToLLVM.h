#pragma once
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tiny {

void populateElementwiseOpToLLVM(LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns);

void populateBinaryOpToLLVM(LLVMTypeConverter &converter,
                            RewritePatternSet &patterns);

void populateTernaryOpToLLVM(LLVMTypeConverter &converter,
                             RewritePatternSet &patterns);

void populateBufferOpToLLVM(LLVMTypeConverter &converter,
                            RewritePatternSet &patterns);

void populateLoadOpToLLVM(LLVMTypeConverter &converter,
                          RewritePatternSet &patterns);

void populateReduceOpToLLVM(LLVMTypeConverter &converter,
                            RewritePatternSet &patterns);

void populateFuncOpToLLVM(LLVMTypeConverter &converter,
                          RewritePatternSet &patterns);
} // namespace mlir::tiny