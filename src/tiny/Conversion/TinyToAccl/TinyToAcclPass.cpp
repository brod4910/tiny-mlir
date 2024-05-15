#include "tiny/Conversion/TinyToAccl/TinyToAcclPass.h"

#define GEN_PASS_DECL_CONVERTTINYTOACCL
#define GEN_PASS_DEF_CONVERTTINYTOACCL
#include "tiny/Conversion/TinyToAccl/Passes.h.inc"

namespace mlir::tiny {
class ConvertTinyToAccl
    : public impl::ConvertTinyToAcclBase<ConvertTinyToAccl> {};
} // namespace mlir::tiny