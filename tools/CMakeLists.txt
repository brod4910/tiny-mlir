get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set (LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRTiny
    MLIRTinyPasses
    MLIRPass
    MLIRTransforms
)

add_llvm_executable(tiny-opt tiny-opt.cpp)

llvm_update_compile_flags(tiny-opt)
target_link_libraries(tiny-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(tiny-opt)