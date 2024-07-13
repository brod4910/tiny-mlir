#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "tiny/Conversion/TinyToAccl/Passes.h"
#include "tiny/Conversion/TinyToLLVM/Passes.h"
#include "tiny/Dialect/Tiny/Transform/Passes.h"
#include "tiny/Dialect/Tiny/Transform/Patterns.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::tiny::TinyDialect>();
  registry.insert<mlir::tiny::accl::AcclDialect>();

  mlir::registerAllDialects(registry);

  mlir::registerAllPasses();
  mlir::tiny::registerTinyPatterns();
  mlir::tiny::registerTinyTransformPasses();
  mlir::tiny::registerTinyToAcclPasses();
  mlir::tiny::registerTinyToLLVMPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tiny Pass Driver", registry));
}