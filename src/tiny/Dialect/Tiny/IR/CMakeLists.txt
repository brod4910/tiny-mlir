set(LLVM_TARGET_DEFINITIONS TinyDialect.td)
mlir_tablegen(TinyDialect.h.inc -gen-dialect-decls -dialect=tiny)
mlir_tablegen(TinyDialect.cpp.inc -gen-dialect-defs -dialect=tiny)
add_public_tablegen_target(TinyDialectIncGen)
add_dependencies(mlir-headers TinyDialectIncGen)

set(LLVM_TARGET_DEFINITIONS TinyOps.td)
mlir_tablegen(TinyOps.h.inc -gen-op-decls)
mlir_tablegen(TinyOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TinyOpIncGen)
add_dependencies(mlir-headers TinyOpIncGen)

set(LLVM_TARGET_DEFINITIONS TinyTypes.td)
mlir_tablegen(TinyTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TinyTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(TinyTypesIncGen)
add_dependencies(mlir-headers TinyTypesIncGen)

set(LLVM_TARGET_DEFINITIONS TinyAttrs.td)
mlir_tablegen(TinyAttrsEnums.h.inc -gen-enum-decls)
mlir_tablegen(TinyAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(TinyAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(TinyAttrsIncGen)
add_dependencies(mlir-headers TinyAttrsIncGen)

add_mlir_doc(TinyDialect TinyDialect Tiny/ -gen-dialect-doc)

add_tiny_library(MLIRTiny
    TinyDialect.cpp
    TinyOps.cpp

    DEPENDS
    TinyDialectIncGen
    TinyOpIncGen
    TinyTypesIncGen
    TinyAttrsIncGen
)