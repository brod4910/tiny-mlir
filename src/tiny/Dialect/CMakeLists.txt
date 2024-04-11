mlir_tablegen(TinyDialect.h.inc -gen-dialect-decls -dialect=tiny)
mlir_tablegen(TinyDialect.cpp.inc -gen-dialect-defs -dialect=tiny)
add_public_tablegen_target(MLIRTinyOpsIncGen)