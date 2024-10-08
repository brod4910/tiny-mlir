#include "tiny/Conversion/TinyToAccl/TinyToAcclPass.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tiny/Dialect/Accelerator/Transform/AcclConversion.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DECL_CONVERTTINYTOACCL
#define GEN_PASS_DEF_CONVERTTINYTOACCL
#include "tiny/Conversion/TinyToAccl/Passes.h.inc"

using namespace mlir;
using namespace mlir::tiny;
using namespace mlir::tiny::accl;

namespace {
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

/*
Generic pattern converter seems pretty widely used across many difference MLIR
Dialects. Most of the Tiny ops can be converted as passthrough and modifyng the
encodings on the RankedTensorTypes.
*/
template <class Op>
struct PassThroughUnaryPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = op->getResult(0).getType();

    // Ranked tensors types are strictly only allowed.
    if (resultType && !llvm::isa<RankedTensorType>(resultType)) {
      return failure();
    }

    auto newResultType = this->getTypeConverter()->convertType(resultType);
    rewriter.replaceOpWithNewOp<Op>(op, newResultType, adaptor.getOperands(),
                                    op->getAttrs());
    return success();
  }
};

template <class Op>
struct PassThroughBinaryPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  // TODO: I believe I need to convert the operands as well.
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = op->getResult(0).getType();

    // Ranked tensors types are strictly only allowed.
    if (resultType && !llvm::isa<RankedTensorType>(resultType)) {
      return failure();
    }

    auto newResultType = this->getTypeConverter()->convertType(resultType);
    rewriter.replaceOpWithNewOp<Op>(op, newResultType, adaptor.getOperands(),
                                    op->getAttrs());
    return success();
  }
};

struct ConstantPattern : public OpConversionPattern<tiny::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tiny::ConstantOp op,
                  typename tiny::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto retShapedType = llvm::cast<ShapedType>(retType);
    auto value = llvm::dyn_cast<DenseElementsAttr>(adaptor.getValue());

    assert(value);
    value = value.reshape(retShapedType);

    addNamedAttrs(
        rewriter.replaceOpWithNewOp<tiny::ConstantOp>(op, retShapedType, value),
        adaptor.getAttributes());
    return success();
  }
};

void populateTinyPatternsAndLegality(AcclTypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  MLIRContext *context = typeConverter.getContext();
  patterns.add<ConstantPattern,
               /* -------- Unary Patterns -------- */
               PassThroughUnaryPattern<tiny::BitcastOp>,
               PassThroughUnaryPattern<tiny::CastOp>,
               PassThroughUnaryPattern<tiny::Exp2Op>,
               PassThroughUnaryPattern<tiny::Log2Op>,
               PassThroughUnaryPattern<tiny::NoOp>,
               PassThroughUnaryPattern<tiny::NegOp>,
               PassThroughUnaryPattern<tiny::RecipOp>,
               PassThroughUnaryPattern<tiny::SinOp>,
               PassThroughUnaryPattern<tiny::SqrtOp>,
               /* -------- Binary Patterns -------- */
               PassThroughBinaryPattern<tiny::AddOp>,
               PassThroughBinaryPattern<tiny::SubOp>,
               PassThroughBinaryPattern<tiny::MulOp>,
               PassThroughBinaryPattern<tiny::DivOp>,
               PassThroughBinaryPattern<tiny::CmpNeOp>,
               PassThroughBinaryPattern<tiny::CmpLtOp>,
               PassThroughBinaryPattern<tiny::MaximumOp>,
               PassThroughBinaryPattern<tiny::ModOp>>(typeConverter, context);
}

class ConvertTinyToAccl
    : public ::impl::ConvertTinyToAcclBase<ConvertTinyToAccl> {
public:
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
    AcclConversionTarget conversionTarget(*context, typeConverter);

    RewritePatternSet patterns(context);

    populateTinyPatternsAndLegality(typeConverter, patterns);
    auto i32_ty = IntegerType::get(module->getContext(), 32);

    module->setAttr(
        "accl.num-warps",
        IntegerAttr::get(i32_ty, llvm::APInt(32, numWarps.getValue())));
    module->setAttr(
        "accl.threads-per-warp",
        IntegerAttr::get(i32_ty, llvm::APInt(32, threadsPerWarp.getValue())));

    if (failed(applyPartialConversion(module, conversionTarget,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tiny::createConvertTinyToAcclPass(std::string target, int numWarps,
                                        int threadsPerWarp) {
  return std::make_unique<::ConvertTinyToAccl>(target, numWarps,
                                               threadsPerWarp);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tiny::createConvertTinyToAcclPass() {
  return std::make_unique<::ConvertTinyToAccl>();
}