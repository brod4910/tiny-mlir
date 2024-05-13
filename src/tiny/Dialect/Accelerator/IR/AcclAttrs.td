#ifndef ACCL_ATTRS
#define ACCL_ATTRS
include "TinyTypes.td"

include "AcclDialect.td"

include "mlir/IR/AttrTypeBase.td"

class Accl_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<Accl_Dialect, name, traits, "::mlir::Attribute"> {
    let description = [{}];

    let attrName = "tiny.accl." # attrMnemonic;
}

// def CTAInfoAttr : Accl_Attr<"CTAInfoAttr", "cta_info"> {
//   let parameters = (ins)
// }

def TensorCoreEncodingAttr : Accl_Attr<"TensorCore", "tensor_core"> {
    let parameters = (ins "unsigned":$MDim,
                          "unsigned":$NDim,
                          "unsigned":$KDim,
                          "Type":$DtypeIn, 
                          "Type":$DtypeOut);

    let hasCustomAssemblyFormat = 1;
}

#endif // ACCL_ATTRS