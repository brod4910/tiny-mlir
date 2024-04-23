#include "TinyPatterns.h"
#include "TinyDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Tools/PDLL/CodeGen/MLIRGen.h"
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>

#include "mlir/Parser/Parser.h"
#include "tiny/Dialect/Tiny/IR/TinyPatterns.inc"

namespace mlir::tiny {
struct EraseNoOp : public PassWrapper<EraseNoOp, OperationPass<>> {
  FrozenRewritePatternSet patterns;

  StringRef getArgument() const final { return "tiny-erase-noop"; }

  StringRef getDescription() const final { return "Tiny Erase NoOp"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TinyDialect>();
  }

  LogicalResult initialize(MLIRContext *ctx) override {
    // Build the pattern set within the `initialize` to avoid recompiling PDL
    // patterns during each `runOnOperation` invocation.
    RewritePatternSet pattern_list(ctx);
    pattern_list.add<EraseNoOpPattern>(ctx);
    patterns = std::move(pattern_list);
    return success();
  }

  void runOnOperation() final {
    (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
  };
};

struct CastNoOp : public PassWrapper<CastNoOp, OperationPass<>> {
  FrozenRewritePatternSet patterns;

  StringRef getArgument() const final { return "tiny-cast-noop"; }

  StringRef getDescription() const final { return "Tiny Cast NoOp"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TinyDialect>();
  }

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet pattern_list(ctx);
    pattern_list.add<CastNoOpPattern>(ctx);
    patterns = std::move(pattern_list);
    return success();
  }

  void runOnOperation() final {
    (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
  }
};

void registerTinyPasses() {
  PassRegistration<EraseNoOp>();
  PassRegistration<CastNoOp>();
}
} // namespace mlir::tiny