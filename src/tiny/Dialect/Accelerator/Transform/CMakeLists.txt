add_mlir_pdll_library(MLIRTinyAcclIncTransforms Passes.pdll Passes.inc EXTRA_INCLUDES
                      ${MLIR_INCLUDE_DIR})


add_tiny_library(MLIRTinyAcclTransforms
    Passes.cpp
    AcclConversion.cpp

    DEPENDS
    MLIRTinyAcclIncTransforms
    
    LINK_LIBS PUBLIC
    MLIRTiny
    MLIRTinyAccl
    )