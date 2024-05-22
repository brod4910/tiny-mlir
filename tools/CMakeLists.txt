get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(tiny_libs GLOBAL PROPERTY TINY_LIBS)

set (LIBS
    MLIROptLib
    MLIRTiny
    MLIRTinyTransforms
    MLIRTinyAccl
    MLIRTinyAcclTransforms
    TinyToAccl
    MLIRPass
    MLIRTransforms
    ${dialect_libs}
    ${conversion_libs}
    ${tiny_libs}
)

add_llvm_executable(tiny-opt tiny-opt.cpp)

llvm_update_compile_flags(tiny-opt)
target_link_libraries(tiny-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(tiny-opt)