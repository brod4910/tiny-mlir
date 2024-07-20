#ifndef TINY_TRANSFORM_PASSES
#define TINY_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def TinyBufferizePass : Pass<"tiny-bufferize", "ModuleOp"> {
    let summary = "Bufferize the Tiny dialect Ops.";
    let description = [{
        This pass bufferizes tiny dialect ops.
    }];
}

def TinyElementwiseToLinalg : Pass<"convert-tiny-elementwise-to-linalg", ""> {
    let summary = "Convert Tiny elementwise ops to Linalg ops";
    let description = [{
        This transforms tiny elementwise ops to linalg ops.
    }];

    let constructor = "tiny::createConvertTinyElementwiseToLinalgPass()";
    let dependentDialects = ["tiny::TinyDialect", "tiny::accl::AcclDialect", 
                             "linalg::LinalgDialect", "scf::SCFDialect"];
}

def ConvertTinyFuncOps : Pass<"convert-tiny-func-ops", ""> {
    let summary = "Simple conversion for Tiny Func Ops to FuncDialect ops.";
    let description = [{
        A simple passthrough conversion for the func ops in the Tiny Dialect to
        the func ops in the Func Dialect.

        This may get removed in the future depending on if Tiny requires special
        function semantics. I see no need ATM (7/19/24).
    }];

    let dependentDialects = ["tiny::TinyDiact", "func::FuncDialect"];
}

#endif //  TINY_TRANSFORM_PASSES