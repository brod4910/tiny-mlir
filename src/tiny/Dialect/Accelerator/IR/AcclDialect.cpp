#include "tiny/Dialect/Accelerator/IR/AcclDialect.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"
#include <functional>

#include "tiny/Dialect/Accelerator/IR/AcclDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tiny/Dialect/Accelerator/IR/AcclAttrs.cpp.inc"

#include <cstdint>
#include <numeric>

namespace mlir::tiny::accl {
/*
---------------------------------------------------
---------------------- DIALECT --------------------
--------------------------------------------------- */
void AcclDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tiny/Dialect/Accelerator/IR/AcclAttrs.cpp.inc"
      >();
}

Operation *AcclDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return tiny::ConstantOp::materialize(builder, value, type, loc);
}

/*
TODO: Move to file
---------------------------------------------------
---------------------- UTILITY --------------------
--------------------------------------------------- */
int ceil_div(int a, int b) { return (a + b - 1) / b; }

static LogicalResult parseIntAttrValue(AsmParser &parser, Attribute attr,
                                       unsigned &value, StringRef desc) {
  auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
  if (!intAttr) {
    parser.emitError(parser.getNameLoc(), "expected an integer type in ")
        << desc;
    return failure();
  }
  if (intAttr.getType().isSignedInteger()) {
    int64_t attrVal = intAttr.getSInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else if (intAttr.getType().isSignlessInteger()) {
    int64_t attrVal = intAttr.getInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else {
    value = intAttr.getUInt();
  }
  return success();
}

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned> &res,
                                       StringRef desc) {
  auto arrayAttr = mlir::dyn_cast<ArrayAttr>(attr.getValue());
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
    return failure();
  }
  for (Attribute i : arrayAttr) {
    unsigned value;
    if (parseIntAttrValue(parser, i, value, desc).failed())
      return failure();
    res.push_back(value);
  }
  return success();
};

/*
---------------------------------------------------
-------------- CTA Layout Attribute ---------------
--------------------------------------------------- */

CTALayoutAttr getDefaultCTALayout(MLIRContext *context,
                                  ArrayRef<int64_t> shape) {
  int rank = shape.size();
  constexpr uint64_t threadsPerWarp = 8;
  uint64_t numel =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  uint64_t numWarps = ceil_div(numel, threadsPerWarp);
  auto threadBlockTile = ThreadBlockTileAttr::get(context, 1, 1, 1);
  auto warpTile = WarpTileAttr::get(context, numWarps, 1, 1);
  auto threadTile = ThreadTileAttr::get(context, threadsPerWarp, 1, 1);

  return CTALayoutAttr::get(context, threadBlockTile, warpTile, threadTile);
}

void CTALayoutAttr::print(AsmPrinter &printer) const {
  auto threadBlockTile = getThreadBlockTile();
  auto warpTile = getWarpTile();
  auto threadTile = getThreadTile();

  printer << "<{"
          << "threadBlockTile = [" << threadBlockTile.getM() << ", "
          << threadBlockTile.getN() << ", " << threadBlockTile.getK() << "]"
          << ", warpTile = [" << warpTile.getM() << ", " << warpTile.getN()
          << ", " << warpTile.getK() << "]"
          << ", threadTile = [" << threadTile.getM() << ", "
          << threadTile.getN() << ", " << threadTile.getK() << "]";

  printer << "}>";
}

Attribute CTALayoutAttr::parse(AsmParser &parser, Type type) {
  DictionaryAttr dict;

  if (parser.parseLess().failed() || parser.parseAttribute(dict).failed() ||
      parser.parseGreater().failed()) {
    return {};
  }

  SmallVector<unsigned> threadBlockTile;
  SmallVector<unsigned> warpTile;
  SmallVector<unsigned> threadTile;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "threadBlockTile") {
      if (parseIntArrayAttr(parser, attr, threadBlockTile,
                            "thread block tile dimensions")
              .failed()) {
        return {};
      }
    } else if (attr.getName() == "warpTile") {
      if (parseIntArrayAttr(parser, attr, warpTile,
                            "thread block tile dimensions")
              .failed()) {
        return {};
      }
    } else if (attr.getName() == "threadTile") {
      if (parseIntArrayAttr(parser, attr, threadTile,
                            "thread block tile dimensions")
              .failed()) {
        return {};
      }
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected attribute key: ")
          << attr.getName().strref();
      return {};
    }
  }

  return parser.getChecked<CTALayoutAttr>(
      parser.getContext(), ArrayRef(threadBlockTile), ArrayRef(warpTile),
      ArrayRef(threadTile));
}

LogicalResult
CTALayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      ThreadBlockTileAttr threadBlockTile,
                      WarpTileAttr warpTile, ThreadTileAttr threadTile) {
  // TODO: Probably need to verify valid CTALayout but can't think of any
  // validation checks atm. 5/23/24
  return success();
}

/*
---------------------------------------------------
----------- Thread Block Tile Attribute -----------
--------------------------------------------------- */

/*
---------------------------------------------------
---------------- Warp Tile Attribute --------------
--------------------------------------------------- */

/*
---------------------------------------------------
------------- Thread Tile Attribute ---------------
--------------------------------------------------- */

/*
---------------------------------------------------
-------------- Tensor Core Attribute --------------
--------------------------------------------------- */

} // namespace mlir::tiny::accl