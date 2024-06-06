#ifndef TINY_ATTRS
#define TINY_ATTRS

include "tiny/Dialect/Tiny/IR/TinyDialect.td"

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/AttrTypeBase.td"

class Tiny_Attr<string name, list<Trait> traits = []>
                : AttrDef<Tiny_Dialect, name, traits> {
  let cppNamespace = "::mlir::tiny";
}

// Note: Syntatic sugar that will get transformed into
// an affine map. Mostly just for ease of use in the IR.
def SliceAttr : Tiny_Attr<"Slice", []> {  
  let parameters = (ins OptionalParameter<"signed">: $start,
                        OptionalParameter<"signed">: $end,
                        OptionalParameter<"signed">: $stride);
  let mnemonic = "slice";
  
  let builders = [
    AttrBuilder<(ins "int64_t":$start,
                     "int64_t":$end,
                     "int64_t":$stride), [{
        return $_get(context, {start}, {end}, {stride});
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";  

}

#endif // TINY_ATTRS