#include "tiny/Conversion/TinyToLLVM/TinyToLLVMPass.h"

#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"
#include <memory>

#define GEN_PASS_DECL_CONVERTTINYTOACCL
#define GEN_PASS_DEF_CONVERTTINYTOACCL
#include "tiny/Conversion/TinyToLLVM/Passes.h.inc"

using namespace mlir;
using namespace mlir::tiny;
using namespace mlir::tiny::accl;

class ConvertTinyToLLVM
    : public ::impl::ConvertTinyToLLVMBase<ConvertTinyToLLVM> {};

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tiny::createConvertTinyToLLVMPass() {
  return std::make_unique<::ConvertTinyToLLVM>();
}