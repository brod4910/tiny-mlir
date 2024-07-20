#include "tiny/Dialect/Tiny/Transform/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"
#include "tiny/Dialect/Tiny/Transform/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
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
#include <sys/signal.h>

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

struct FuncOpPattern : PatternRewriter {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tiny::FuncOp op, tiny::FuncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<func::FuncOp>(
        op, op.getName(), op.getFunctionType());
    auto converter = getTypeConverter();

    addNamedAttrs(newOp, adaptor.getAttributes());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), *converter)))
      return failure();

    return success();
  }
};

class CallOpPattern : public OpConversionPattern<tiny::CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tiny::CallOp op, tiny::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, op.getCallee(), op.getResultTypes(), adaptor.getOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());
    return success();
  }
};

class ReturnOpPattern : public OpConversionPattern<tiny::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tiny::ReturnOp op, tiny::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

void populateTinyFuncOpPatterns(RewritePatternSet &patterns) {
  patterns.add<FuncOpPattern, ReturnOpPattern, CallOpPattern>();
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

    populateTinyFuncOpPatterns(patterns);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tiny::createConvertTinyElementwiseToLinalgPass() {
  return std::make_unique<TinyElementwiseToLinalgPass>();
}

// std::unique_ptr<Pass> mlir::tiny::createConvertTinyFuncOps() {
//   return std::make_unique<ConvertTinyFuncOps>();
// }