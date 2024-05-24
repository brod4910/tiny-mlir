#ifndef ACCL_DIALECT
#define ACCL_DIALECT

include "mlir/IR/DialectBase.td"

def Accl_Dialect : Dialect {
    let name = "accl";

    let summary = "An MLIR for general hardware-agnostic accelerator optimizations";

    let description = [{
        An MLIR for hardware-independent optimizations. Essentially a middle-layer dialect
        that provides mid-level transformations on the IR.

        This dialect gets lowered to hardware-specific dialects where low-level optimiztions can
        be performed.
    }];

    let dependentDialects = [
        "tiny::TinyDialect",
        "scf::SCFDialect",
    ];

    let cppNamespace = "::mlir::tiny::accl";

    let hasConstantMaterializer = 1;

    // TODO: Use when we have types on Accl Dialect?
    // let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let usePropertiesForAttributes = 1;
}

#endif // ACCL_DIALECT