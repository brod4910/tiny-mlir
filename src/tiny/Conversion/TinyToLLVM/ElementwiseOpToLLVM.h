#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

namespace mlir::tiny {
using Exp2OpLowering = VectorConvertToLLVMPattern<Exp2Op, LLVM::Exp2Op>;
using Log2OpLowering = VectorConvertToLLVMPattern<Log2Op, LLVM::Log2Op>;
// using CastOpLowering = VectorConvertToLLVMPattern<CastOp, LLVM::Cast>
using BitcastOpLowering =
    VectorConvertToLLVMPattern<BitcastOp, LLVM::BitcastOp>;
using SinOpLowering = VectorConvertToLLVMPattern<SinOp, LLVM::SinOp>;
using SqrtOpLowering = VectorConvertToLLVMPattern<SqrtOp, LLVM::SqrtOp>;
// using NegOpLowering = VectorConvertToLLVMPattern<NegOp, >
// using RecipOpLowering = VectorConvertToLLVMPattern<RecipOp,
} // namespace mlir::tiny