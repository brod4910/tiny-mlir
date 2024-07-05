#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "tiny/Dialect/Tiny/IR/TinyDialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace mlir::tiny {
/*
To stay as close to the tinygrad front-end as possible, this section has a
lot of repeated code that could be circumvented by just adding Float and Int
variants of the ops. No fun in that though :)
*/
TypedAttr getConstantAttr(Type type, int64_t value, PatternRewriter &rewriter) {
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    Type eTy = shapedTy.getElementType();
    APInt valueInt(eTy.getIntOrFloatBitWidth(), value);
    return DenseIntElementsAttr::get(shapedTy, valueInt);
  }

  return rewriter.getIntegerAttr(type, value);
}

template <typename SourceOp, template <typename, typename>
                             typename AttrConvert = AttrConvertPassThrough>
class NegOpLowering : public ConvertOpToLLVMPattern<SourceOp> {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) {
    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");
    auto value = op.getValue();
    auto valueType = value.getType();
    auto eleType = valueType.getElementType();

    llvm::StringRef targetOp;

    if (llvm::isa<FloatType>(eleType)) {
      AttrConvert<SourceOp, LLVM::FNegOp> attrConvert(op);

      return LLVM::detail::vectorOneToOneRewrite(
          op, LLVM::FNegOp::getOperationName(), adaptor.getOperands(),
          attrConvert.getAttrs(), *this->getTypeConverter(), rewriter);
    } else if (llvm::isa<IntegerType>(eleType)) {
      AttrConvert<SourceOp, LLVM::SubOp> attrConvert(op);

      auto zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), getConstantAttr(valueType, 0, rewriter));

      return LLVM::detail::vectorOneToOneRewrite(
          op, LLVM::SubOp::getOperationName(), {zero, value},
          attrConvert.getAttrs(), *this->getTypeConverter(), rewriter);
    }
  }
};

using Log2OpLowering = VectorConvertToLLVMPattern<Log2Op, LLVM::Log2Op>;
// using CastOpLowering = VectorConvertToLLVMPattern<CastOp, LLVM::Cast>
using BitcastOpLowering =
    VectorConvertToLLVMPattern<BitcastOp, LLVM::BitcastOp>;
using SinOpLowering = VectorConvertToLLVMPattern<SinOp, LLVM::SinOp>;
using SqrtOpLowering = VectorConvertToLLVMPattern<SqrtOp, LLVM::SqrtOp>;
// using NegOpLowering = VectorConvertToLLVMPattern<NegOp, >
// using RecipOpLowering = VectorConvertToLLVMPattern<RecipOp,
} // namespace mlir::tiny