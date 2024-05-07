add_mlir_pdll_library(MLIRTinyIncPasses Passes.pdll Passes.inc EXTRA_INCLUDES
                      ${MLIR_INCLUDE_DIR})


add_mlir_dialect_library(MLIRTinyPasses
    Passes.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/src
    
    LINK_LIBS PUBLIC
    )