#include "tiny/Conversion/TinyToAccl/TinyToAcclPass.h"
#include "tiny/Dialect/Accelerator/Transform/AcclConversion.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#define GEN_PASS_DECL_CONVERTTINYTOACCL
#define GEN_PASS_DEF_CONVERTTINYTOACCL
#include "tiny/Conversion/TinyToAccl/Passes.h.inc"

namespace mlir::tiny {
class ConvertTinyToAccl
    : public ::impl::ConvertTinyToAcclBase<ConvertTinyToAccl> {
  ConvertTinyToAccl() = default;

  ConvertTinyToAccl(const std::string target, int numWarps = 4,
                    int threadsPerWarp = 32) {
    this->target = target;
    this->numWarps = numWarps;
    this->threadsPerWarp = threadsPerWarp;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    AcclTypeConverter typeConverter(context, numWarps, threadsPerWarp);

    RewritePatternSet patterns(context);
  }
};
} // namespace mlir::tiny