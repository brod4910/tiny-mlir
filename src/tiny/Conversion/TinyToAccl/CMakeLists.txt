set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TinyToAccl)
add_public_tablegen_target(TinyConversionPassIncGen)

add_tiny_library(TinyToAccl
    TinyToAcclPass.cpp

    DEPENDS
    TinyConversionPassIncGen
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRTiny
)