#include "tiny/Dialect/Tiny/Transform/Patterns.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Tools/PDLL/CodeGen/MLIRGen.h"
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>

#include "mlir/Parser/Parser.h"

#include "tiny/Dialect/Tiny/Transform/Patterns.h.inc"

namespace mlir::tiny {
struct RemoveRedundantOps
    : public PassWrapper<RemoveRedundantOps, OperationPass<>> {

  FrozenRewritePatternSet patterns;

  StringRef getArgument() const final { return "tiny-remove-redundant"; }

  StringRef getDescription() const final {
    return "Remove Redundant Tiny Ops.";
  }

  LogicalResult initialize(MLIRContext *ctx) override {
    // Build the pattern set within the `initialize` to avoid recompiling PDL
    // patterns during each `runOnOperation` invocation.
    RewritePatternSet patternList(ctx);
    patternList.add<CastNoOpPattern>(ctx);
    patternList.add<EraseNoOpPattern>(ctx);
    patterns = std::move(patternList);
    return success();
  }

  void runOnOperation() override {
    if (applyPatternsAndFoldGreedily(getOperation(), patterns).failed()) {
      signalPassFailure();
    }
  }
};

void registerTinyPasses() { PassRegistration<RemoveRedundantOps>(); }
} // namespace mlir::tiny