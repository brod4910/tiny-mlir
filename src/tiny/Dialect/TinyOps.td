#ifndef TINY_OPS
#define TINY_OPS

include "src/tiny/Dialect/TinyDialect.td"
include "mlir/IR/OpBase.td"

class Tiny_Op <string mnemonic, list<Trait> traits = []> : Op<Tiny_Dialect, mnemonic, traits>;


#endif // TINY_OPS