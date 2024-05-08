add_mlir_pdll_library(MLIRTinyIncTransforms Passes.pdll Passes.inc EXTRA_INCLUDES
                      ${MLIR_INCLUDE_DIR})


add_tiny_library(MLIRTinyTransforms
    Passes.cpp

    DEPENDS
    MLIRTinyIncTransforms
    
    LINK_LIBS PUBLIC
    MLIRTiny
    )