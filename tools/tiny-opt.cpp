#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"
#include "tiny/Dialect/Tiny/Transform/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::tiny::TinyDialect>();
  registry.insert<mlir::tiny::accl::AcclDialect>();
  mlir::registerAllDialects(registry);

  mlir::registerAllPasses();
  mlir::tiny::registerTinyPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tiny Pass Driver", registry));
}