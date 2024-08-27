#include "tiny/Conversion/TinyToLLVM/TinyAttrConverter.h"

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
} // namespace mlir::tiny