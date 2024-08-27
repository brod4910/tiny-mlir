#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

#include "tiny/Dialect/Tiny/IR/TinyDialect.h"
#include <utility>

namespace mlir::tiny {
LLVM::FastmathFlags convertTinyFastmathFlagsToLLVM(
    FastmathFlags TFMF = FastmathFlags{FastmathFlags::fast});
} // namespace mlir::tiny