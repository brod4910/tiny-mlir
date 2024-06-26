#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "tiny/Conversion/TinyToAccl/Passes.h"
#include "tiny/Dialect/Tiny/Transform/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::registerAllPasses();
  mlir::tiny::registerTinyPasses();
  mlir::tiny::registerTinyToAcclPasses();

  registry.insert<mlir::tiny::TinyDialect>();
  registry.insert<mlir::tiny::accl::AcclDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tiny Pass Driver", registry));
}