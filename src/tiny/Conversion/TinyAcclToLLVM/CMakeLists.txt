set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TinyAcclToLLVM)
add_public_tablegen_target(TinyAcclToLLVMConversionPassIncGen)

add_tiny_library(TinyAcclToNVGPU
    TinyAcclToLLVM.cpp

    DEPENDS
    TinyAcclToLLVMConversionPassIncGen
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRTiny
    MLIRTinyAccl
)