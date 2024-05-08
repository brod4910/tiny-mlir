set(LLVM_TARGET_DEFINITIONS AcclDialect.td)
mlir_tablegen(AcclDialect.h.inc -gen-dialect-decls -dialect=accl)
mlir_tablegen(AcclDialect.cpp.inc -gen-dialect-defs -dialect=accl)
add_public_tablegen_target(AcclIRIncGen)
add_dependencies(mlir-headers AcclIRIncGen)

add_mlir_doc(AcclDialect AcclDialect Accelerator/ -gen-dialect-doc)

add_mlir_dialect_library(MLIRAccl
    AcclDialect.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src

    DEPENDS
    MLIRTiny
    MLIRSCFDialect
    
    LINK_LIBS PUBLIC
    )