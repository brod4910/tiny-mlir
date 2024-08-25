#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::tiny {
// MFLAGS = ('nsz', 'arcp', 'contract', 'afn', 'reassoc') # All from fast math,
// but nnan and ninf

LLVM::FastmathFlags TinyFastmathFlags{
    LLVM::FastmathFlags::nsz | LLVM::FastmathFlags::arcp |
    LLVM::FastmathFlags::contract | LLVM::FastmathFlags::afn |
    LLVM::FastmathFlags::reassoc};

} // namespace mlir::tiny