#ifndef TINY_CONVERSION_PASSES
#define TINY_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTinyToAccl : Pass<"convert-tiny-accl-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Tiny Accl to LLVM";
    let description = [{}];

    let constructor = "mlir::tiny::createConvertTinyAcclToLLVMPass()";

    let dependentDialects = ["mlir::tiny::TinyDialect",
                             "mlir::tiny::accl::AcclDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [];

}

#endif // TINY_CONVERSION_PASSES
