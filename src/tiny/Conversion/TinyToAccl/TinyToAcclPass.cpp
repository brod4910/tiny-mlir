#include "tiny/Conversion/TinyToAccl/TinyToAcclPass.h"
#include "tiny/Dialect/Accelerator/Transform/AcclConversion.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#define GEN_PASS_DECL_CONVERTTINYTOACCL
#define GEN_PASS_DEF_CONVERTTINYTOACCL
#include "tiny/Conversion/TinyToAccl/Passes.h.inc"

using namespace mlir;
using namespace mlir::tiny;
using namespace mlir::tiny::accl;

namespace {
/*
Generic pattern converter seems pretty widely used across many difference MLIR
Dialects. Most of the Tiny ops can be converted as passthrough and modifyng the
encodings on the RankedTensorTypes.
*/
template <class Op>
struct PassThroughUnaryPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) {
    auto resultType = op->getResult()->getType();
    if (!resultType && !llvm::isa<RankedTensorType>(resultType)) {
      return failure();
    }

    auto newResultType = this->getTypeConverter()->convertType(resultType);
    rewriter.replaceOpWithNewOp<Op>(op, newResultType, adaptor.getOperands(),
                                    op->getAttrs());
    return success();
  }
};

void populateTinyPatternsAndLegality(AcclTypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  MLIRContext *context = typeConverter.getContext();
  patterns.add<PassThroughUnaryPattern<tiny::Exp2Op>,
               PassThroughUnaryPattern<tiny::NoOp>,
               PassThroughUnaryPattern<tiny::SinOp>,
               PassThroughUnaryPattern<tiny::Log2Op>,
               PassThroughUnaryPattern<tiny::CastOp>,
               PassThroughUnaryPattern<tiny::SqrtOp>,
               PassThroughUnaryPattern<tiny::NegOp>>(typeConverter, context);
}

class ConvertTinyToAccl
    : public ::impl::ConvertTinyToAcclBase<ConvertTinyToAccl> {
  ConvertTinyToAccl() = default;

  ConvertTinyToAccl(const std::string target, int numWarps = 4,
                    int threadsPerWarp = 32) {
    this->target = target;
    this->numWarps = numWarps;
    this->threadsPerWarp = threadsPerWarp;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    AcclTypeConverter typeConverter(context, numWarps, threadsPerWarp);

    RewritePatternSet patterns(context);
  }
};
} // namespace
