include "mlir/Pass/PassBase.td"

def TinyInferShapes : Pass<"tiny-infer-shapes", "tiny::FuncOp"> {
    let summary = "Propagate shapes across Tiny operations";
    let description = [{
        Pass that uses operand types and propagates shapes to Tiny operations.
        This includes legalizing rankless and dynamic shapes towards static.
    }];

    let constructor = "createTinyInferShapesPass";
    
    let dependentDialects = [
        "tiny::TinyDialect"
    ];
}