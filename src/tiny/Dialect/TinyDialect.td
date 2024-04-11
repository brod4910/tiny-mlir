#ifndef TINY_DIALECT
#define TINY_DIALECT

include "mlir/IR/DialectBase.td"

def Tiny_Dialect : Dialect {
    let name = "tiny";
}
#endif // TINY_DIALECT