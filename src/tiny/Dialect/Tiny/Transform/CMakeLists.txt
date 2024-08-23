add_mlir_pdll_library(MLIRTinyIncPatternTransforms Patterns.pdll Patterns.h.inc EXTRA_INCLUDES
                      ${MLIR_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TinyTransform)
add_public_tablegen_target(MLIRTinyIncPassesTransforms)



add_tiny_library(MLIRTinyTransforms
    Passes.cpp
    Patterns.cpp
    BufferizableOpInterfaceImpl.cpp

    DEPENDS
    MLIRTinyIncPassesTransforms
    MLIRTinyIncPatternTransforms
    
    LINK_LIBS PUBLIC
    MLIRTiny
    )