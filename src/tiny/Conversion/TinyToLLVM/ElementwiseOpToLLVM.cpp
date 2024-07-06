#include "tiny/Conversion/TinyToLLVM/ElementwiseOpToLLVM.h"
#include "tiny/Conversion/TinyToLLVM/PatternTinyToLLVM.h"

namespace mlir::tiny {
void populateElementwiseOpToLLVM(LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns) {
  patterns.add<BitcastOpLowering, Exp2OpLowering, Log2OpLowering, NegOpLowering,
               RecipOpLowering, SinOpLowering, SqrtOpLowering>(converter);
}
} // namespace mlir::tiny