#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

namespace mlir::tiny {
static constexpr StringRef barePtrAttrName = "llvm.bareptr";

/// Return `true` if the `op` should use bare pointer calling convention.
static bool shouldUseBarePtrCallConv(Operation *op,
                                     const LLVMTypeConverter *typeConverter) {
  return (op && op->hasAttr(barePtrAttrName)) ||
         typeConverter->getOptions().useBarePtrCallConv;
}

class FuncOpLowering : public ConvertOpToLLVMPattern<func::FuncOp> {
  using ConvertOpToLLVMPattern<func::FuncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, func::FuncOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::LLVMFuncOp newFuncOp =
        *convertFuncOpToLLVMFuncOp(cast<FunctionOpInterface>(op.getOperation()),
                                   rewriter, *getTypeConverter());

    if (!newFuncOp) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// Special lowering pattern for `ReturnOps`.  Unlike all other operations,
// `ReturnOp` interacts with the function signature and must have as many
// operands as the function has return values.  Because in LLVM IR, functions
// can only return 0 or 1 value, we pack multiple values into a structure type.
// Emit `UndefOp` followed by `InsertValueOp`s to create such structure if
// necessary before returning it
struct ReturnOpLowering : public ConvertOpToLLVMPattern<ReturnOp> {
  using ConvertOpToLLVMPattern<ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned numArguments = op.getNumOperands();
    SmallVector<Value, 4> updatedOperands;

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    bool useBarePtrCallConv =
        shouldUseBarePtrCallConv(funcOp, this->getTypeConverter());
    if (useBarePtrCallConv) {
      // For the bare-ptr calling convention, extract the aligned pointer to
      // be returned from the memref descriptor.
      for (auto it : llvm::zip(op->getOperands(), adaptor.getOperands())) {
        Type oldTy = std::get<0>(it).getType();
        Value newOperand = std::get<1>(it);
        if (isa<MemRefType>(oldTy) && getTypeConverter()->canConvertToBarePtr(
                                          cast<BaseMemRefType>(oldTy))) {
          MemRefDescriptor memrefDesc(newOperand);
          newOperand = memrefDesc.allocatedPtr(rewriter, loc);
        } else if (isa<UnrankedMemRefType>(oldTy)) {
          // Unranked memref is not supported in the bare pointer calling
          // convention.
          return failure();
        }
        updatedOperands.push_back(newOperand);
      }
    } else {
      updatedOperands = llvm::to_vector<4>(adaptor.getOperands());
      (void)copyUnrankedDescriptors(rewriter, loc, op.getOperands().getTypes(),
                                    updatedOperands,
                                    /*toDynamic=*/true);
    }

    // If ReturnOp has 0 or 1 operand, create it and return immediately.
    if (numArguments <= 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
          op, TypeRange(), updatedOperands, op->getAttrs());
      return success();
    }

    // Otherwise, we need to pack the arguments into an LLVM struct type before
    // returning.
    auto packedType = getTypeConverter()->packFunctionResults(
        op.getOperandTypes(), useBarePtrCallConv);
    if (!packedType) {
      return rewriter.notifyMatchFailure(op, "could not convert result types");
    }

    Value packed = rewriter.create<LLVM::UndefOp>(loc, packedType);
    for (auto [idx, operand] : llvm::enumerate(updatedOperands)) {
      packed = rewriter.create<LLVM::InsertValueOp>(loc, packed, operand, idx);
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), packed,
                                                op->getAttrs());
    return success();
  }
};
} // namespace mlir::tiny