#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "tiny/Conversion/TinyToLLVM/TinyAttrConverter.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace mlir::tiny {
/*
Simple Lowering patterns that don't have multiple type versions of the same Op
or that require special treatment when converting to LLMV.
*/
using Log2OpToLLVM = VectorConvertToLLVMPattern<Log2Op, LLVM::Log2Op>;
using BitcastOpToLLVM = VectorConvertToLLVMPattern<BitcastOp, LLVM::BitcastOp>;
using SinOpToLLVM = VectorConvertToLLVMPattern<SinOp, LLVM::SinOp>;
using SqrtOpToLLVM = VectorConvertToLLVMPattern<SqrtOp, LLVM::SqrtOp>;
using Exp2OpToLLVM = VectorConvertToLLVMPattern<Exp2Op, LLVM::Exp2Op>;

// TODO: implement Cast properly
// using CastOpLowering = VectorConvertToLLVMPattern<CastOp, LLVM::Cast>

/*
To stay as close to the tinygrad front-end as possible, this section has
duplicate codels that could be circumvented by just adding Float and Int
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

template <typename SourceOp, typename Op1, typename Op2,
          typename T1 = FloatType, typename T2 = IntegerType>
LogicalResult selectOpFromElementType(SourceOp op, Type elementType,
                                      llvm::StringLiteral &operationName) {
  if (llvm::isa<T1>(elementType)) {
    operationName = Op1::getOperationName();
    return success();
  } else if (llvm::isa<T2>(elementType)) {
    operationName = Op2::getOperationName();
    return success();
  } else {
    return emitError(op->getLoc(), "Element type should be one of: "
                                   "IntegerType or FloatType but got: ")
           << elementType;
  }
}

class NegOpToLLVM : public ConvertOpToLLVMPattern<NegOp> {
  using ConvertOpToLLVMPattern<NegOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(NegOp op, NegOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static_assert(std::is_base_of<OpTrait::OneResult<NegOp>, NegOp>::value,
                  "expected single result op");
    auto value = op.getValue();
    auto valueType = value.getType();
    auto eleType = getElementTypeOrSelf(valueType);

    auto llvmFmf = convertTinyFastmathFlagsToLLVM();
    auto fmfAttr = LLVM::FastmathFlagsAttr::get(op.getContext(), llvmFmf);

    auto fmfNamedAttr =
        rewriter.getNamedAttr(LLVM::FastmathFlagsAttr::name, fmfAttr);

    if (llvm::isa<FloatType>(eleType)) {
      return LLVM::detail::vectorOneToOneRewrite(
          op, LLVM::FNegOp::getOperationName(), adaptor.getOperands(),
          {fmfNamedAttr}, *getTypeConverter(), rewriter);
    } else if (llvm::isa<IntegerType>(eleType)) {
      /*
      LLVM doesn't have an explicit INegOp so we create a sub 0, val Op.
      Also mentioned in the LLVM docs for reference.
      */

      auto zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), getConstantAttr(valueType, 0, rewriter));

      auto res = LLVM::detail::vectorOneToOneRewrite(
          op, LLVM::SubOp::getOperationName(), ValueRange{zero, value},
          {fmfNamedAttr}, *getTypeConverter(), rewriter);

      if (res.failed()) {
        emitRemark(op->getLoc(), "Failed Int Neg");
      }
      return res;
    } else {
      return emitError(op->getLoc(), "Element type should be one of: "
                                     "IntegerType or FloatType but got: ")
             << eleType;
    }
  }
};

class RecipOpToLLVM : public ConvertOpToLLVMPattern<RecipOp> {
  using ConvertOpToLLVMPattern<RecipOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RecipOp op, RecipOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Check if dtype is of IntegerType? Pretty sure this will fail
    // probably need to either insert cast before FDiv, or just use IDiv and
    // assume user wants IDiv
    auto value = op.getValue();
    auto valueType = value.getType();

    auto llvmFmf = convertTinyFastmathFlagsToLLVM();
    auto fmfAttr = LLVM::FastmathFlagsAttr::get(op.getContext(), llvmFmf);

    auto fmfNamedAttr =
        rewriter.getNamedAttr(LLVM::FastmathFlagsAttr::name, fmfAttr);

    auto one = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), getConstantAttr(valueType, 1, rewriter));

    AttrConvertPassThrough<RecipOp, LLVM::FDivOp> attrConvert(op);
    return LLVM::detail::vectorOneToOneRewrite(
        op, LLVM::FDivOp::getOperationName(), ValueRange{one, value},
        {fmfNamedAttr}, *getTypeConverter(), rewriter);
  }
};

template <typename SourceOp, typename FloatOp, typename IntegerOp>
class GenericBinaryOpToLLVMPattern : public ConvertOpToLLVMPattern<SourceOp> {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = op->getResult(0);
    auto resultType = result.getType();
    auto resultElementType = getElementTypeOrSelf(resultType);

    llvm::StringLiteral operationName{""};

    if (selectOpFromElementType<SourceOp, FloatOp, IntegerOp>(
            op, resultElementType, operationName)
            .failed()) {
      return failure();
    }

    auto llvmFMF = getTinyDefaultLLVMFastmathFlags(op->getContext(), rewriter);
    return LLVM::detail::vectorOneToOneRewrite(
        op, operationName, adaptor.getOperands(), {llvmFMF},
        *this->getTypeConverter(), rewriter);
  }
};

class CmpLtOpToLLVM : public ConvertOpToLLVMPattern<CmpLtOp> {
  using ConvertOpToLLVMPattern<CmpLtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CmpLtOp op, CmpLtOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = op->getResult(0);
    auto resultType = result.getType();
    auto resultElementType = getElementTypeOrSelf(resultType);

    llvm::StringLiteral operationName{""};
  }
};

} // namespace mlir::tiny