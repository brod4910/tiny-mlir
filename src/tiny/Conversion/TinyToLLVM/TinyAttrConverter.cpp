#include "tiny/Conversion/TinyToLLVM/TinyAttrConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Attributes.h"

namespace mlir::tiny {
LLVM::FastmathFlags convertTinyFastmathFlagsToLLVM(FastmathFlags TFMF) {
  LLVM::FastmathFlags llvmFMF;

  const std::pair<FastmathFlags, LLVM::FastmathFlags> flags[] = {
      {
          FastmathFlags::afn,
          LLVM::FastmathFlags::afn,
      },
      {
          FastmathFlags::arcp,
          LLVM::FastmathFlags::arcp,
      },
      {
          FastmathFlags::contract,
          LLVM::FastmathFlags::contract,
      },
      {
          FastmathFlags::nsz,
          LLVM::FastmathFlags::nsz,
      },
      {
          FastmathFlags::reassoc,
          LLVM::FastmathFlags::reassoc,
      },
      {
          FastmathFlags::fast,
          {LLVM::FastmathFlags::afn | LLVM::FastmathFlags::arcp |
           LLVM::FastmathFlags::contract | LLVM::FastmathFlags::nsz |
           LLVM::FastmathFlags::reassoc},
      },
  };

  for (auto [tfmf, llvmFmf] : flags) {
    if (tiny::bitEnumContainsAny(tfmf, TFMF)) {
      llvmFMF = llvmFMF | llvmFmf;
    }
  }

  return llvmFMF;
}

LLVM::FastmathFlagsAttr
getTinyDefaultLLVMFastmathFlagsAttr(MLIRContext *context,
                                    ConversionPatternRewriter &rewriter) {
  auto llvmFmf = convertTinyFastmathFlagsToLLVM();
  auto fmfAttr = LLVM::FastmathFlagsAttr::get(context, llvmFmf);

  return fmfAttr;
}

NamedAttribute
getTinyDefaultLLVMFastmathFlagsNamedAttr(MLIRContext *context,
                                         ConversionPatternRewriter &rewriter) {
  auto fmfAttr = getTinyDefaultLLVMFastmathFlagsAttr(context, rewriter);
  return rewriter.getNamedAttr(LLVM::FastmathFlagsAttr::name, fmfAttr);
}

} // namespace mlir::tiny