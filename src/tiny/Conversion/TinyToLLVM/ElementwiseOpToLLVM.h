#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "tiny/Conversion/TinyToLLVM/TinyAttrConverter.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace mlir::tiny {
/*
Simple Lowering patterns that don't have multiple type versions of the same Op
or that require special treatment when converting to LLMV.
*/

/*
---------------------------------------------------
------------------- Unary Ops ---------------------
--------------------------------------------------- */
using Log2OpToLLVM = VectorConvertToLLVMPattern<Log2Op, LLVM::Log2Op>;
using BitcastOpToLLVM = VectorConvertToLLVMPattern<BitcastOp, LLVM::BitcastOp>;
using SinOpToLLVM = VectorConvertToLLVMPattern<SinOp, LLVM::SinOp>;
using SqrtOpToLLVM = VectorConvertToLLVMPattern<SqrtOp, LLVM::SqrtOp>;
using Exp2OpToLLVM = VectorConvertToLLVMPattern<Exp2Op, LLVM::Exp2Op>;
// TODO: implement Cast properly
// using CastOpLowering = VectorConvertToLLVMPattern<CastOp, LLVM::Cast>

/*
---------------------------------------------------
------------------ Binary Ops ---------------------
--------------------------------------------------- */
using MaximumOpToLLVM = VectorConvertToLLVMPattern<MaximumOp, LLVM::MaximumOp>;
using XOROpToLLVM = VectorConvertToLLVMPattern<XOROp, LLVM::XOrOp>;
// TODO: Explore Logical Shr operand as well?
using ShrOpToLLVM = VectorConvertToLLVMPattern<ShrOp, LLVM::AShrOp>;
using ShlOpToLLV = VectorConvertToLLVMPattern<ShlOp, LLVM::ShlOp>;
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

template <typename TargetOp, typename OpAdaptor, typename... Args>
LogicalResult vectorOneToOneRewrite(Operation *op, OpAdaptor adaptor,
                                    const LLVMTypeConverter &typeConverter,
                                    ConversionPatternRewriter &rewriter,
                                    Args... args) {
  if (!llvm::all_of(op->getOperandTypes(), LLVM::isCompatibleType)) {
    return failure();
  }

  auto resultType = op->getResult(0).getType();

  if (!llvm::isa<LLVM::LLVMArrayType>(adaptor.getLhs().getType())) {
    rewriter.replaceOpWithNewOp<TargetOp>(
        op, typeConverter.convertType(resultType), args...);
    return success();
  }

  if (!isa<VectorType>(resultType))
    return rewriter.notifyMatchFailure(op, "expected vector result type");

  auto callback = [&](Type llvm1DVectorTy, ValueRange operands) {
    return rewriter.create<TargetOp>(op->getLoc(), llvm1DVectorTy, args...);
  };

  return LLVM::detail::handleMultidimensionalVectors(
      op, op->getOperands(), typeConverter, callback, rewriter);
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

    if (llvm::isa<FloatType>(eleType)) {
      auto llvmFmf =
          getTinyDefaultLLVMFastmathFlagsNamedAttr(op.getContext(), rewriter);
      return LLVM::detail::vectorOneToOneRewrite(
          op, LLVM::FNegOp::getOperationName(), adaptor.getOperands(),
          {llvmFmf}, *getTypeConverter(), rewriter);
    } else if (llvm::isa<IntegerType>(eleType)) {
      /*
      LLVM doesn't have an explicit INegOp so we create a sub 0, val Op.
      Also mentioned in the LLVM docs for reference.
      */

      auto zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), getConstantAttr(valueType, 0, rewriter));

      auto res = LLVM::detail::vectorOneToOneRewrite(
          op, LLVM::SubOp::getOperationName(), ValueRange{zero, value}, {},
          *getTypeConverter(), rewriter);

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

    auto llvmFmf =
        getTinyDefaultLLVMFastmathFlagsNamedAttr(op.getContext(), rewriter);

    auto one = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), getConstantAttr(valueType, 1, rewriter));

    AttrConvertPassThrough<RecipOp, LLVM::FDivOp> attrConvert(op);
    return LLVM::detail::vectorOneToOneRewrite(
        op, LLVM::FDivOp::getOperationName(), ValueRange{one, value}, {llvmFmf},
        *getTypeConverter(), rewriter);
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

    auto llvmFMF =
        getTinyDefaultLLVMFastmathFlagsNamedAttr(op->getContext(), rewriter);
    return LLVM::detail::vectorOneToOneRewrite(
        op, operationName, adaptor.getOperands(), {llvmFMF},
        *this->getTypeConverter(), rewriter);
  }
};

template <typename SourceOp, LLVM::FCmpPredicate FPredicate,
          LLVM::ICmpPredicate IPredicate>
class GenericCmpOpToLLVMPattern : public ConvertOpToLLVMPattern<SourceOp> {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = op.getLhs().getType();
    auto elementType = getElementTypeOrSelf(operandType);

    if (llvm::isa<FloatType>(elementType)) {
      auto predicate =
          LLVM::FCmpPredicateAttr::get(op.getContext(), FPredicate);
      auto llvmFMF =
          getTinyDefaultLLVMFastmathFlagsAttr(op.getContext(), rewriter);

      return vectorOneToOneRewrite<LLVM::FCmpOp, OpAdaptor,
                                   LLVM::FCmpPredicateAttr, Value, Value,
                                   LLVM::FastmathFlagsAttr>(
          op, adaptor, *this->getTypeConverter(), rewriter, predicate,
          adaptor.getLhs(), adaptor.getRhs(), llvmFMF);
    } else if (llvm::isa<IntegerType>(elementType)) {
      auto predicate =
          LLVM::ICmpPredicateAttr::get(op.getContext(), IPredicate);
      return vectorOneToOneRewrite<LLVM::ICmpOp, OpAdaptor,
                                   LLVM::ICmpPredicateAttr, Value, Value>(
          op, adaptor, *this->getTypeConverter(), rewriter, predicate,
          adaptor.getLhs(), adaptor.getRhs());
    } else {
      return emitError(op->getLoc(), "Element type should be one of: "
                                     "IntegerType or FloatType but got: ")
             << elementType;
    }
  }
};
} // namespace mlir::tiny