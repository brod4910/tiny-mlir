#include "tiny/Dialect/Tiny/Transform/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
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
static bool isElementwiseMappableOpOnRankedTensors(Operation *op) {
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return false;

  // TODO: The conversion pattern can be made to work for `any_of` here, but
  // it's more complex as it requires tracking which operands are scalars.
  return llvm::all_of(op->getOperandTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); });
}

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

    linalg::populateElementwiseToLinalgConversionPatterns(patterns);
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !isElementwiseMappableOpOnRankedTensors(op);
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