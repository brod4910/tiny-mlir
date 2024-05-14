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

def ThreadBlockTileAttr : Accl_Attr<"ThreadBlockTileAttr", "thread_block_tile"> {

}

def WarpTileAttr : Accl_Attr<"WarpTileAttr", "warp_tile"> {

}

def ThreadTileAttr : Accl_Attr<"ThreadTileAttr", "thread_tile"> {

}

def TensorCoreAttr : Accl_Attr<"TensorCoreAttr", "tensor_core"> {
      let parameters = (ins "unsigned":$MDim,
                            "unsigned":$NDim,
                            "unsigned":$KDim,
                            "Type":$DtypeIn, 
                            "Type":$DtypeOut);
}

class GEMMEncoding<string name, string attrMnemonic, list<Trait> traits = []> : Accl_Attr<name, attrMnemonic, traits, Accl_Dialect> {
  let description = [{
    A GEMM encoding described by Thread-Block Tiles, Warp Tiles, Thread Tiles, and Tensor Cores.
  }];

  code extraGEMMDeclaration = [{}];
}

#endif // ACCL_ATTRS