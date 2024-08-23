#include "tiny/Conversion/TinyToLLVM/FuncOpToLLVM.h"
#include "tiny/Conversion/TinyToLLVM/PatternTinyToLLVM.h"

namespace mlir::tiny {
void populateFuncOpToLLVM(LLVMTypeConverter &converter,
                          RewritePatternSet &patterns) {
  patterns.add<FuncOpLowering, ReturnOpLowering>(converter);
}
} // namespace mlir::tiny