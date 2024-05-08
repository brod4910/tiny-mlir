set(LLVM_TARGET_DEFINITIONS TinyOps.td)
mlir_tablegen(TinyOps.h.inc -gen-op-decls)
mlir_tablegen(TinyOps.cpp.inc -gen-op-defs)
mlir_tablegen(TinyTypes.h.inc -gen-typedef-decls -typedefs-dialect=tiny)
mlir_tablegen(TinyTypes.cpp.inc -gen-typedef-decls -typedefs-dialect=tiny)
mlir_tablegen(TinyDialect.h.inc -gen-dialect-decls -dialect=tiny)
mlir_tablegen(TinyDialect.cpp.inc -gen-dialect-defs -dialect=tiny)
add_public_tablegen_target(TinyIRIncGen)
add_dependencies(mlir-headers TinyIRIncGen)

add_mlir_doc(TinyDialect TinyDialect Tiny/ -gen-dialect-doc)

add_tiny_library(MLIRTiny
    TinyDialect.cpp
    TinyOps.cpp

    DEPENDS
    TinyIRIncGen
)