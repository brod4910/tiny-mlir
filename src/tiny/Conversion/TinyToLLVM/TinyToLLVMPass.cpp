#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "tiny/Conversion/TinyToLLVM/PatternTinyToLLVM.h"
#include "tiny/Conversion/TinyToLLVM/TinyToLLVMPass.h"
#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include <memory>

#define GEN_PASS_DECL_CONVERTTINYTOLLVM
#define GEN_PASS_DEF_CONVERTTINYTOLLVM
#include "tiny/Conversion/TinyToLLVM/Passes.h.inc"

using namespace mlir;
using namespace mlir::tiny;
using namespace mlir::tiny::accl;

namespace {
class TinyLLVMConversionTarget : public ConversionTarget {
public:
  explicit TinyLLVMConversionTarget(MLIRContext &context)
      : ConversionTarget(context) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();

    addIllegalDialect<tiny::TinyDialect>();
    addIllegalDialect<tiny::accl::AcclDialect>();
    addIllegalDialect<memref::MemRefDialect>();
    addIllegalDialect<affine::AffineDialect>();

    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class ConvertTinyToLLVM
    : public ::impl::ConvertTinyToLLVMBase<ConvertTinyToLLVM> {
public:
  ConvertTinyToLLVM() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    LowerToLLVMOptions options(context);
    options.overrideIndexBitwidth(32);

    LLVMTypeConverter typeConverter(context, options);
    TinyLLVMConversionTarget target(*context);

    RewritePatternSet patterns(context);
    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateElementwiseOpToLLVM(typeConverter, patterns);
    populateFuncOpToLLVM(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tiny::createConvertTinyToLLVMPass() {
  return std::make_unique<::ConvertTinyToLLVM>();
}