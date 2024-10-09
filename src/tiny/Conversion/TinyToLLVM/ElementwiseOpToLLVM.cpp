#include "tiny/Conversion/TinyToLLVM/ElementwiseOpToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
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
      // Two's complement so unsigned/signed the same
      GenericBinaryOpToLLVMPattern<MulOp, LLVM::FMulOp, LLVM::MulOp,
                                   LLVM::MulOp>,
      // Two's complement so unsigned/signed the same
      GenericBinaryOpToLLVMPattern<AddOp, LLVM::FAddOp, LLVM::AddOp,
                                   LLVM::AddOp>,
      // Two's complement so unsigned/signed the same
      GenericBinaryOpToLLVMPattern<SubOp, LLVM::FSubOp, LLVM::SubOp,
                                   LLVM::SubOp>,
      GenericBinaryOpToLLVMPattern<DivOp, LLVM::FDivOp, LLVM::SDivOp,
                                   LLVM::UDivOp>,
      GenericCmpOpToLLVMPattern<CmpNeOp, LLVM::FCmpPredicate::une,
                                LLVM::ICmpPredicate::ne,
                                LLVM::ICmpPredicate::ne>,
      GenericCmpOpToLLVMPattern<CmpLtOp, LLVM::FCmpPredicate::ult,
                                LLVM::ICmpPredicate::slt,
                                LLVM::ICmpPredicate::ult>,
      MaximumOpToLLVM,
      GenericBinaryOpToLLVMPattern<ModOp, LLVM::FRemOp, LLVM::SRemOp,
                                   LLVM::URemOp>,
      XOROpToLLVM, ShrOpToLLVM, ShlOpToLLVM>(converter);
}
} // namespace mlir::tiny