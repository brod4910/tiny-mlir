#ifndef TINY_TRANSFORM_PASSES
#define TINY_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def TinyBufferizePass : Pass<"tiny-bufferize", "ModuleOp"> {
    let summary = "Bufferize the Tiny dialect Ops.";
    let description = [{
        This pass bufferizes tiny dialect ops.
    }];
}

#endif //  TINY_TRANSFORM_PASSES