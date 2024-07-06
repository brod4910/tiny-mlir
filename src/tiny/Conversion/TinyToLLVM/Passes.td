#ifndef TINY_CONVERSION_PASSES
#define TINY_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTinyToLLVM : Pass<"convert-tiny-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Tiny to LLVM";
    let description = [{}];

    let constructor = "mlir::tiny::createConvertTinyToLLVMPass()";

    let dependentDialects = ["mlir::tiny::TinyDialect",
                             "mlir::tiny::accl::AcclDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::memref::MemRefDialect"];
}

#endif // TINY_CONVERSION_PASSES
