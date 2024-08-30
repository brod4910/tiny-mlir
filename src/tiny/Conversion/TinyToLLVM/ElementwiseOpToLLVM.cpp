#include "tiny/Conversion/TinyToLLVM/ElementwiseOpToLLVM.h"
#include "tiny/Conversion/TinyToLLVM/PatternTinyToLLVM.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

namespace mlir::tiny {
void populateElementwiseOpToLLVM(LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns) {
  patterns.add<
      /* -------- Unary Patterns -------- */
      BitcastOpToLLVM, Exp2OpToLLVM, Log2OpToLLVM, NegOpToLLVM, RecipOpToLLVM,
      SinOpToLLVM, SqrtOpToLLVM,
      /* -------- Binary Patterns -------- */
      GenericBinaryOpToLLVMPattern<MulOp, LLVM::FMulOp, LLVM::MulOp>,
      GenericBinaryOpToLLVMPattern<AddOp, LLVM::FAddOp, LLVM::AddOp>,
      GenericBinaryOpToLLVMPattern<SubOp, LLVM::FSubOp, LLVM::SubOp>,
      // TODO: Support UDivOp as well might need to be a non-generic Pattern
      GenericBinaryOpToLLVMPattern<DivOp, LLVM::FDivOp, LLVM::SDivOp>>(
      converter);
}
} // namespace mlir::tiny