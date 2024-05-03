llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  lit.site.cfg.py.in
  lit.cfg.py
)

set(TINY_TEST_DEPENDS
  tiny-opt
)

set(FILECHECK_PATH "${LLVM_LIBRARY_DIR}/../bin/FileCheck")
set(LIT_ARGS "-Dfilecheck=${FILECHECK_PATH}")
add_lit_testsuite(check-tiny-lit-tests "Running the Tiny regression tests"
  ${CMAKE_BINARY_DIR}
  ARGS ${LIT_ARGS}
  DEPENDS ${TINY_TEST_DEPENDS}
  )

set_target_properties(check-tiny-lit-tests PROPERTIES FOLDER "Tests")

add_lit_testsuites(TINY-LIT-TESTS ${CMAKE_SOURCE_DIR} DEPENDS ${TINY_TEST_DEPENDS})