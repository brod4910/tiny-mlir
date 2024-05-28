#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"

#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include <optional>

namespace mlir::tiny {
/*
---------------------------------------------------
------------------- CONSTANT OP -------------------
--------------------------------------------------- */
bool ConstantOp::verifyWith(Attribute value, Type type) {
  auto rankedType = llvm::dyn_cast<RankedTensorType>(type);

  if (!rankedType) {
    return false;
  }

  if (llvm::isa<IntegerType>(rankedType) &&
      !llvm::cast<IntegerType>(type).isSignless()) {
    return false;
  }

  if (!llvm::isa<DenseIntOrFPElementsAttr>(value)) {
    return false;
  }

  return true;
}

ConstantOp ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) {
  if (verifyWith(value, type)) {
    return builder.create<ConstantOp>(loc, type,
                                      llvm::cast<ElementsAttr>(value));
  }

  return nullptr;
}

LogicalResult ConstantOp::verify() {
  auto type = llvm::dyn_cast<RankedTensorType>(getType());

  if (!type) {
    return emitOpError("value must be ranked tensor.");
  }

  if (llvm::isa<IntegerType>(type) &&
      !llvm::cast<IntegerType>(type).isSignless()) {
    return emitOpError("integer return type must be signless.");
  }

  if (!llvm::isa<DenseIntOrFPElementsAttr>(getValue())) {
    return emitOpError(
        "value must be integer or floating-point elements attributes.");
  }

  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

/*
---------------------------------------------------
------------------- UNARY OPS ---------------------
--------------------------------------------------- */

/* ------------------ CastOp ------------------- */

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return false;
  }

  auto input = llvm::dyn_cast<TensorType>(inputs.front());
  auto output = llvm::dyn_cast<TensorType>(outputs.front());

  if (!input || !output) {
    return false;
  }

  // The shape is required to match if both types are ranked.
  return succeeded(verifyCompatibleShape(input, output));
}

/*
---------------------------------------------------
------------------- BINARY OPS --------------------
--------------------------------------------------- */

TypeRange padLeft(TypeRange types, int64_t maxRank) {
  SmallVector<Type> padded;

  for (auto type : types) {
    auto rankedType = cast<RankedTensorType>(type);

    if (rankedType.getRank() == maxRank) {
      padded.push_back(rankedType);
    }

    int64_t pad = maxRank - rankedType.getRank();
    auto shape = SmallVector<int64_t>(pad, 1);

    for (auto d : rankedType.getShape()) {
      shape.push_back(d);
    }

    auto newRankType =
        RankedTensorType::get(shape, rankedType.getElementType(), {});
    padded.push_back(newRankType);
  }
  return padded;
}

/* ------------------ Add Op ---------------------- */

LogicalResult AddOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    llvm::SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  Value lhs = operands.front();
  Value rhs = operands.back();

  auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());

  if (!lhsType || !rhsType) {
    return failure();
  }

  SmallVector<int64_t, 4> resultShape;
  if (!OpTrait::util::getBroadcastedShape(lhsType.getShape(),
                                          rhsType.getShape(), resultShape)) {
    return failure();
  }

  inferredReturnShapes.emplace_back(resultShape);

  return success();
}

/*
---------------------------------------------------
------------------ FUNCTION OPS -------------------
---------------------------------------------------

             COPIED FROM THE MLIR REPO
*/

/* ------------------ FuncOp ------------------- */

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

/* ------------------ CallOp ------------------- */

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

/* ------------------ ReturnOp ------------------- */

LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}

} // namespace mlir::tiny