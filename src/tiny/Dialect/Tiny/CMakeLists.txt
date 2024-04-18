add_mlir_pdll_library(MLIRTinyPatterns TinyPatterns.pdll TinyPatterns.h.inc)

set(LLVM_TARGET_DEFINITIONS TinyOps.td)
mlir_tablegen(TinyOps.h.inc -gen-op-decls)
mlir_tablegen(TinyOps.cpp.inc -gen-op-defs)
mlir_tablegen(TinyDialect.h.inc -gen-dialect-decls -dialect=tiny)
mlir_tablegen(TinyDialect.cpp.inc -gen-dialect-defs -dialect=tiny)
add_public_tablegen_target(TinyOpsIncGen)
add_dependencies(mlir-headers TinyOpsIncGen)

add_mlir_doc(TinyDialect TinyDialect Tiny/ -gen-dialect-doc)

add_mlir_dialect_library(MLIRTiny
    TinyDialect.cpp
    TinyOps.cpp
    TinyPatterns.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src/tiny/Dialect/Tiny

    DEPENDS
    MLIRTinyPatterns

    LINK_LIBS PUBLIC
    )