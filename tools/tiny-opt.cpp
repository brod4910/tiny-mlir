#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "tiny/Dialect/Tiny/TinyDialect.h"
#include "tiny/Dialect/Tiny/TinyPatterns.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::tiny::TinyDialect, mlir::arith::ArithDialect,
                  mlir::math::MathDialect>();

  mlir::registerAllPasses();
  mlir::tiny::registerTinyPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tiny Pass Driver", registry));
}