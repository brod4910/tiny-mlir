#include "TinyPatterns.h"
#include "TinyDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "tiny/Dialect/Tiny/IR/TinyPatterns.h.inc"

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
    RewritePatternSet patternList(ctx);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void runOnOperation() final {
    // Invoke the pattern driver with the provided patterns.
    (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
  }
};

void registerTinyPasses() { PassRegistration<EraseNoOp>(); }

} // namespace mlir::tiny