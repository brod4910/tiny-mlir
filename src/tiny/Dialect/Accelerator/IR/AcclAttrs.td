#ifndef ACCL_ATTRS
#define ACCL_ATTRS

include "tiny/Dialect/Tiny/IR/TinyTypes.td"

include "AcclDialect.td"

include "mlir/IR/AttrTypeBase.td"

class Accl_Attr<string name, string attrMnemonic, list<Trait> traits = []>
                : AttrDef<Accl_Dialect, name, traits, "::mlir::Attribute"> {
  let cppNamespace = "::mlir::tiny::accl";
  let description = [{}];
  let mnemonic = "accl." # attrMnemonic;
}

class Accl_DimensionAttr<string name, string attrMnemonic, list<Trait> traits = []> 
                         : Accl_Attr<name, attrMnemonic, traits> {
  let parameters = (ins "unsigned": $M,
                        "unsigned": $N,
                        "unsigned": $K);

  let assemblyFormat = "`<` struct(params) `>`";
}

def ThreadBlockTileAttr : Accl_DimensionAttr<"ThreadBlockTile", "thread_block_tile">;

def WarpTileAttr : Accl_DimensionAttr<"WarpTile", "warp_tile">;

def ThreadTileAttr : Accl_DimensionAttr<"ThreadTile", "thread_tile">;

def CTALayoutAttr : Accl_Attr<"CTALayout", "cta_layout"> {
  let parameters = (ins "ThreadBlockTileAttr":$threadBlockTile,
                        "WarpTileAttr":$warpTile,
                        "ThreadTileAttr":$threadTile);

  let builders = [
    AttrBuilder<(ins "ArrayRef<unsigned>":$threadBlockTile,
                     "ArrayRef<unsigned>":$warpTile,
                     "ArrayRef<unsigned>":$threadTile), [{
        auto threadBlockTileAttr = ThreadBlockTileAttr::get(context, threadBlockTile[0], threadBlockTile[1], threadBlockTile[2]);
        auto warpTileAttr = WarpTileAttr::get(context, warpTile[0], warpTile[1], warpTile[2]);
        auto threadTileAttr = ThreadTileAttr::get(context, threadTile[0], threadTile[1], threadTile[2]);

        return $_get(context, threadBlockTileAttr, warpTileAttr, threadTileAttr);
    }]>
  ];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def TensorCoreAttr : Accl_DimensionAttr<"TensorCore", "tensor_core"> {
  let parameters = (ins "unsigned":$M,
                        "unsigned":$N,
                        "unsigned":$K,
                        "Type":$DtypeIn, 
                        "Type":$DtypeOut);
}

// class GEMMEncoding<string name, string attrMnemonic, list<Trait> traits = []> : Accl_Attr<name, attrMnemonic, traits> {
//   let description = [{
//     A GEMM encoding described by Thread-Block Tiles, Warp Tiles, Thread Tiles, and Tensor Cores.
//   }];

//   code extraGEMMDeclaration = [{
//     SmallVector<unsigned> getThreadBlockTileOrder();
//     SmallVector<unsigned> getWarpTileOrder();
//     SmallVector<unsigned> getThreadTileOrder();
//   }];
// }

// def AmpereGEMMEncodingAttr : GEMMEncoding<"AmpereGEMMEncoding", "ampere_gemm_encoding"> {
//   let mnemonic = "ampere_gemm";

//   let description = [{
//     Encoding that describes valid a GEMM design for the Ampere architecture
//   }];

//   let parameters = (ins "CTALayoutAttr":$ctaLayout,
//                         "TensorCoreAttr":$tensorCore);

//   let extraClassDeclaration = extraGEMMDeclaration;
//   let hasCustomAssemblyFormat = 1;
// }

#endif // ACCL_ATTRS