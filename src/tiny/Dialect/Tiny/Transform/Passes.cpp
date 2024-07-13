#include "tiny/Dialect/Tiny/Transform/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"
#include "tiny/Dialect/Tiny/Transform/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
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

    bufferization::BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    /*
      Function Op patterns (func, call, return)
    */
    populateFunctionOpInterfaceTypeConversionPattern<tiny::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<tiny::FuncOp>([&](tiny::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<tiny::CallOp>(
        [&](tiny::CallOp op) { return typeConverter.isLegal(op); });

    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    target.addLegalOp<ModuleOp, bufferization::ToTensorOp,
                      bufferization::ToMemrefOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }

    OneShotBufferizationOptions options;

    if (failed(runOneShotBufferize(module, options))) {
      signalPassFailure();
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
} // namespace

std::unique_ptr<Pass> mlir::tiny::createConvertTinyElementwiseToLinalgPass() {
  return std::make_unique<TinyElementwiseToLinalgPass>();
}