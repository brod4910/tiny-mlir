#include "tiny/Conversion/TinyToAccl/TinyToAcclPass.h"

#define GEN_PASS_DEF_CONVERTTINYTOACCL
#include "tiny/Conversion/TinyToAccl/Passes.h.inc"

namespace mlir::tiny {
class ConvertTInyToAccl
    : public impl::ConvertTinyToAcclBase<ConvertTInyToAccl> {};
} // namespace mlir::tiny