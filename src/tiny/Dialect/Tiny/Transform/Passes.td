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

#endif //  TINY_TRANSFORM_PASSES