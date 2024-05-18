set(LLVM_TARGET_DEFINITIONS AcclDialect.td)
mlir_tablegen(AcclDialect.h.inc -gen-dialect-decls -dialect=accl)
mlir_tablegen(AcclDialect.cpp.inc -gen-dialect-defs -dialect=accl)
add_public_tablegen_target(AcclIRIncGen)
add_dependencies(mlir-headers AcclIRIncGen)

set(LLVM_TARGET_DEFINITIONS AcclAttrs.td)
mlir_tablegen(AcclAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(AcclAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(AcclAttrsIncGen)
add_dependencies(mlir-headers AcclAttrsIncGen)

add_mlir_doc(AcclDialect AcclDialect Accelerator/ -gen-dialect-doc)
add_mlir_doc(AcclAttrs AcclAttrs Accelerator/ -gen-dialect-doc)

add_tiny_library(MLIRTinyAccl
    AcclDialect.cpp

    DEPENDS
    AcclIRIncGen
    AcclAttrsIncGen
    
    LINK_LIBS PUBLIC
    MLIRTiny
    MLIRSCFDialect
    )