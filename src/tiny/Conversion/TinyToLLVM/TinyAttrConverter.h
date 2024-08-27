#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

#include "tiny/Dialect/Tiny/IR/TinyDialect.h"
#include <utility>

namespace mlir::tiny {
LLVM::FastmathFlags convertTinyFastmathFlagsToLLVM(
    FastmathFlags TFMF = FastmathFlags{FastmathFlags::fast});

NamedAttribute
getTinyDefaultLLVMFastmathFlags(MLIRContext *context,
                                ConversionPatternRewriter &rewriter);

} // namespace mlir::tiny