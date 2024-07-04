#pragma once

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class TinyToLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  using mlir::TypeConverter::convertType;

  TinyToLLVMTypeConverter(mlir::MLIRContext *ctx,
                          mlir::LowerToLLVMOptions &options,
                          const mlir::DataLayoutAnalysis *analysis = nullptr);

  mlir::Type convertTensorType(mlir::RankedTensorType type);
};

} // namespace mlir
