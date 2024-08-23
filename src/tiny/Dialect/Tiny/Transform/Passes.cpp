#include "tiny/Dialect/Tiny/Transform/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"
#include "tiny/Dialect/Tiny/Transform/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include <_types/_uint64_t.h>
#include <memory>

namespace mlir {
namespace tiny {
#define GEN_PASS_DEF_TINYBUFFERIZEPASS
#define GEN_PASS_DEF_TINYELEMENTWISETOLINALG
#define GEN_PASS_DEF_CONVERTTINYFUNCOPS
#include "tiny/Dialect/Tiny/Transform/Passes.h.inc"
} // namespace tiny
} // namespace mlir

using namespace mlir;
using namespace bufferization;

namespace {
struct TinyBufferizePass
    : public tiny::impl::TinyBufferizePassBase<TinyBufferizePass> {
  using TinyBufferizePassBase::TinyBufferizePassBase;
  void runOnOperation() final {
    auto module = getOperation();
    auto *context = &getContext();

    BufferizationOptions options = getPartialBufferizationOptions();
    options.opFilter.allowDialect<tiny::TinyDialect>();

    if (failed(bufferizeOp(getOperation(), options))) {
      return signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect,
                    tiny::TinyDialect>();
    tiny::registerBufferizableOpInterfaceExternalModels(registry);
  }
};

struct ConvertAnyElementwiseBroadcastableOpToLinalg : public RewritePattern {
  ConvertAnyElementwiseBroadcastableOpToLinalg(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    if (!tiny::isElementwiseBroadcastableOpOnRankedTensors(op)) {
      return rewriter.notifyMatchFailure(
          op, "requires elementwise broadcastable on ranked tensors.");
    }

    auto resultType = dyn_cast<RankedTensorType>(op->getResultTypes().front());
    auto rank = resultType.getRank();

    Value resultTensor = rewriter.create<tiny::EmptyOp>(
        op->getLoc(), resultType.getShape(), resultType.getElementType());

    // Get AffineMaps for inputs, check for 1 in dim for broadcasting
    auto affineMaps =
        llvm::map_to_vector(op->getOperands(), [&](Value operand) {
          auto shape = cast<ShapedType>(operand.getType()).getShape();
          SmallVector<AffineExpr> affineExprs;
          for (auto it : llvm::enumerate(shape)) {
            auto expr = it.value() == 1 ? rewriter.getAffineConstantExpr(0)
                                        : rewriter.getAffineDimExpr(it.index());
            affineExprs.push_back(expr);
          }
          return AffineMap::get(rank, 0, affineExprs, rewriter.getContext());
        });

    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // AffineMap for
    affineMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

    // Transform tensor operation to scalar operation here
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, /*resultTensorTypes=*/op->getResultTypes(),
        /*inputs=*/op->getOperands(), /*outputs=*/resultTensor,
        /*indexingMaps=*/affineMaps, /*iteratorTypes=*/iteratorTypes,
        /*bodyBuilder=*/
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_back(op->getNumOperands()),
                             {resultType.getElementType()}, op->getAttrs());
          builder.create<linalg::YieldOp>(loc, scalarOp->getResults());
        });

    return success();
  }
};

TypedAttr selectInitialValue(Operation *op, Type elementType,
                             PatternRewriter &rewriter) {
  if (dyn_cast<tiny::SumOp>(op) && isa<IntegerType>(elementType))
    return rewriter.getIntegerAttr(elementType, 0);

  if (dyn_cast<tiny::SumOp>(op) && isa<FloatType>(elementType))
    return rewriter.getFloatAttr(elementType, 0.0);

  if (dyn_cast<tiny::MaximumOp>(op) && isa<IntegerType>(elementType))
    return rewriter.getIntegerAttr(
        elementType,
        APInt::getSignedMaxValue(elementType.getIntOrFloatBitWidth()));

  if (dyn_cast<tiny::MaximumOp>(op) && isa<FloatType>(elementType))
    return rewriter.getFloatAttr(
        elementType,
        APFloat::getLargest(cast<FloatType>(elementType).getFloatSemantics(),
                            true));

  return {};
}

template <typename SourceOp>
struct ConvertReduceOpToLinalg : public OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const final {
    auto valueType = cast<ShapedType>(op->getOperand(0).getType());
    auto resultType = cast<ShapedType>(op->getResult(0).getType());

    auto axis = op.getAxis();

    // TODO: make this a function on the Op
    // Get real axis. Rank + (-axis) = real axis
    axis = axis >= 0 ? axis : valueType.getRank() + axis;

    Value resultTensor = rewriter.create<tiny::EmptyOp>(
        op->getLoc(), resultType.getShape(), resultType.getElementType());

    auto initialValue =
        selectInitialValue(op, resultType.getElementType(), rewriter);

    if (!initialValue)
      return rewriter.notifyMatchFailure(
          op, "initial value does not exist for reduction op.");

    // Note: Specific to Tiny. Requires everything to be tensors even 0-D
    auto constantTensor =
        rewriter
            .create<tiny::ConstantOp>(
                op->getLoc(),
                DenseElementsAttr::get(
                    RankedTensorType::get({}, resultType.getElementType()),
                    initialValue))
            .getResult();

    auto filledTensor = rewriter
                            .create<tiny::FillOp>(op->getLoc(), resultType,
                                                  resultTensor, constantTensor)
                            .getOperation()
                            ->getResults()
                            .front();

    SmallVector<AffineExpr> resultExpr;
    SmallVector<utils::IteratorType> iteratorTypes;

    for (int d = 0; d < valueType.getRank(); ++d) {
      if (d == axis) {
        iteratorTypes.push_back(utils::IteratorType::reduction);
      } else {
        iteratorTypes.push_back(utils::IteratorType::parallel);
        resultExpr.push_back(rewriter.getAffineDimExpr(d));
      }
    }

    SmallVector<AffineMap, 2> affineMaps = {
        rewriter.getMultiDimIdentityMap(valueType.getRank()),
        AffineMap::get(valueType.getRank(), 0, resultExpr, op.getContext())};

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperand(0), filledTensor, affineMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          Value scalarOp = rewriter.create<tiny::AddOp>(
              loc, resultType.getElementType(), regionArgs[0], regionArgs[1]);
          builder.create<linalg::YieldOp>(loc, scalarOp);
        });

    return success();
  }
};

void populateElementwiseBroadcastableToLinalg(RewritePatternSet &patterns) {
  patterns.add<ConvertAnyElementwiseBroadcastableOpToLinalg,
               ConvertReduceOpToLinalg<tiny::SumOp>,
               ConvertReduceOpToLinalg<tiny::MaxOp>>(patterns.getContext());
}

struct TinyElementwiseToLinalgPass
    : public tiny::impl::TinyElementwiseToLinalgBase<
          TinyElementwiseToLinalgPass> {
  using tiny::impl::TinyElementwiseToLinalgBase<
      TinyElementwiseToLinalgPass>::TinyElementwiseToLinalgBase;

  void runOnOperation() final {
    auto *func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    populateElementwiseBroadcastableToLinalg(patterns);
    linalg::populateElementwiseToLinalgConversionPatterns(patterns);
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !tiny::isElementwiseBroadcastableOpOnRankedTensors(op) &&
             !tiny::isElementwiseOpOnRankedTensors(op) &&
             !tiny::isReducerOpOnRankedTensors(op);
    });

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

struct FuncOpPattern : OpRewritePattern<tiny::FuncOp> {
  FuncOpPattern(MLIRContext *context)
      : OpRewritePattern<tiny::FuncOp>(context) {}

  LogicalResult matchAndRewrite(tiny::FuncOp op,
                                PatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<func::FuncOp>(
        op, op.getName(), op.getFunctionType());

    addNamedAttrs(newOp, op->getAttrDictionary());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    return success();
  }
};

class CallOpPattern : public OpRewritePattern<tiny::CallOp> {
public:
  CallOpPattern(MLIRContext *context)
      : OpRewritePattern<tiny::CallOp>(context) {}

  LogicalResult matchAndRewrite(tiny::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, op.getCallee(), op.getResultTypes(), op->getOperands());
    addNamedAttrs(newOp, op->getAttrDictionary());
    return success();
  }
};

class ReturnOpPattern : public OpRewritePattern<tiny::ReturnOp> {
public:
  ReturnOpPattern(MLIRContext *context)
      : OpRewritePattern<tiny::ReturnOp>(context) {}

  LogicalResult matchAndRewrite(tiny::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.getOperands());
    return success();
  }
};

void populateTinyFuncOpPatterns(MLIRContext *context,
                                RewritePatternSet &patterns) {
  patterns.add<FuncOpPattern, ReturnOpPattern, CallOpPattern>(context);
}

struct ConvertTinyFuncOps
    : public tiny::impl::ConvertTinyFuncOpsBase<ConvertTinyFuncOps> {
  using tiny::impl::ConvertTinyFuncOpsBase<
      ConvertTinyFuncOps>::ConvertTinyFuncOpsBase;

  void runOnOperation() final {
    auto *func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<func::FuncDialect>();

    populateTinyFuncOpPatterns(context, patterns);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace