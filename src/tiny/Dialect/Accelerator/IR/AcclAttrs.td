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
  let parameters = (ins "unsigned": $N,
                        "unsigned": $M,
                        "unsigned": $K);
}

def WarpTileAttr : Accl_Attr<"WarpTileAttr", "warp_tile"> {
  let parameters = (ins "unsigned": $N,
                        "unsigned": $M,
                        "unsigned": $K);
}

def ThreadTileAttr : Accl_Attr<"ThreadTileAttr", "thread_tile"> {
  let parameters = (ins "unsigned": $N,
                        "unsigned": $M,
                        "unsigned": $K);
}

def TensorCoreAttr : Accl_Attr<"TensorCoreAttr", "tensor_core"> {
  let parameters = (ins "unsigned":$M,
                        "unsigned":$N,
                        "unsigned":$K,
                        "Type":$DtypeIn, 
                        "Type":$DtypeOut);
}

class GEMMEncoding<string name, string attrMnemonic, list<Trait> traits = []> : Accl_Attr<name, attrMnemonic, traits, Accl_Dialect> {
  let description = [{
    A GEMM encoding described by Thread-Block Tiles, Warp Tiles, Thread Tiles, and Tensor Cores.
  }];

  code extraGEMMDeclaration = [{

  }];
}

def AmpereGEMMEncoding : AmpereGEMMEncoding<"AmpereGEMMEncoding", "ampere_gemm_encoding"> {
  let mnemonic = "ampere_gemm";

  let description = [{
    Encoding that describes valid a GEMM design for the Ampere architecture
  }];

  let parameters = (ins 
  "ThreadBlockTileAttr":$thread_block_tile,
  "WarpTileAttr":$warp_tile,
  "ThreadTileAttr":$thread_tile,
  "TensorCoreAttr":$tensor_core
  )
}

#endif // ACCL_ATTRS