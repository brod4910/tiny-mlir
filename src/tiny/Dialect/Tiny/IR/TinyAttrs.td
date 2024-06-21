#ifndef TINY_ATTRS
#define TINY_ATTRS

include "tiny/Dialect/Tiny/IR/TinyDialect.td"

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/AttrTypeBase.td"

class Tiny_Attr<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute">
                : AttrDef<Tiny_Dialect, name, traits, baseCppClass> {
  let cppNamespace = "::mlir::tiny";
}

#endif // TINY_ATTRS