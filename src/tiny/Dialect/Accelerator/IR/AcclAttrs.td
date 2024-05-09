#ifndef ACCL_ATTRS
#define ACCL_ATTRS
include "AcclDialect.td"

include "mlir/IR/AttrTypeBase.td"

class Accl_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<Accl_Dialect, name, traits, "::mlir::Attribute"> {
    let description = [{}];

    let attrName = "tiny.accl." # attrMnemonic;
  }


def TensorCoreEncodingAttr : Accl_Attr<"TensorCore", "tensor_core"> {
    let parameters = (ins "unsigned":$MDim, "unsigned":$NDim, "unsigned":$KDim);
}

#endif // ACCL_ATTRS