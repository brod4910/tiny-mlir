set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TinyToLLVM)
add_public_tablegen_target(TinyToLLVMConversionPassIncGen)

add_tiny_library(TinyToLLVM
    TinyToLLVMPass.cpp
    ElementwiseOpToLLVM.cpp
    
    DEPENDS
    TinyToLLVMConversionPassIncGen
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRTiny
    MLIRTinyAccl
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
)