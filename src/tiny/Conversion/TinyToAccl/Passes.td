#ifndef TINY_CONVERSION_PASSES
#define TINY_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTinyToAccl : Pass<"convert-tiny-to-accl", "mlir::ModuleOp"> {
    let summary = "Convert Tiny to Accl";
    let description = [{}];

    let constructor = "mlir::tiny::createConvertTinyToAcclPass()";

    let dependentDialects = ["mlir::tiny::TinyDialect",
                             "mlir::tiny::accl::AcclDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::cf::ControlFlowDialect",
                             "mlir::func::FuncDialect"];

    // TODO: Compute blocks should be the method for general computation
    // abstraction.
    // I.E:
    //  tiny.accl.compute-block<MxNxKx...>
    // this should be attached to either all tensors as an encoding or 
    // memref as a layout or is this more of an Operation level attribute?
    let options = [
        Option<"numWarps", "num-warps", "int32_t", "4", "number of warps">,
        Option<"threadsPerWarp", "threads-per-warp", "int32_t", "32", "number of threads per warp">,
        Option<"target", "target", "std::string", "\"\"", "GPU target">,
    ];

}

#endif // TINY_CONVERSION_PASSES
