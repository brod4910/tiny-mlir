set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TinyOpt)
add_public_tablegen_target(MLIRTinyPassIncGen)
add_dependencies(mlir-headers MLIRTinyPassIncGen)

add_mlir_doc(Passes TinyPasses ./ -gen-pass-doc)