#ifndef TINY_ATTRS
#define TINY_ATTRS

include "tiny/Dialect/Tiny/IR/TinyDialect.td"
include "tiny/Dialect/Tiny/IR/TinyTypes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/AttrTypeBase.td"

class Tiny_Attr<string name, string attrMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute">
                : AttrDef<Tiny_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
  let cppNamespace = "::mlir::tiny";
  let attrName = "tiny." # attrMnemonic;
}

def Tiny_SlicedLayoutAttr : Tiny_Attr<"SlicedLayout", "sliced_layout", [DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface>]> {
  let summary = "An attribute representing a sliced layout of a shaped type.";
  let description = [{
    
  }];

  let parameters = (ins
    ArrayRefParameter<"SliceType", "array of slices">:$slices
  );
}
#endif // TINY_ATTRS