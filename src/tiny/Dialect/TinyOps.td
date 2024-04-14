#ifndef TINY_OPS
#define TINY_OPS

include "TinyDialect.td"
include "TinyTypes.td"

include "mlir/IR/OpBase.td"

class Tiny_Op <string mnemonic, list<Trait> traits = []> : Op<Tiny_Dialect, mnemonic, traits>;

/* 
--------------------------------------------------- 
-------------------- Traits -----------------------
--------------------------------------------------- */

// TODO: Find out why ElementwiseMappable doesn't like being put in the NativeOpTrait

def UnaryTrait : NativeOpTrait<"Unary", []>;

/* 
--------------------------------------------------- 
------------------- UNARY OPS ---------------------
--------------------------------------------------- */

def NoOp : Tiny_Op<"noop", [UnaryTrait, ElementwiseMappable]> {
    let summary = "A No-Op.";

    let description = [{
        A No-Op operation. Undecided but this operation will most likely
        get cannonicalized and removed from the graph.
    }];
}

def Exp2 : Tiny_Op<"exp2", [UnaryTrait, ElementwiseMappable]> {
    let summary = "Two raised by value.";

    let description = [{
        An operation that raises 2 by the value.
    }];
}

def Log2 : Tiny_Op<"log2", [UnaryTrait, ElementwiseMappable]> {
    let summary = "Binary (base-2) logarithm of the value.";

    let description = [{
        An operation that computes the binary (base-2) logarithm of the value.
    }];
}

def Cast : Tiny_Op<"cast", [UnaryTrait, ElementwiseMappable]> {
    let summary = "Cast the value to another type.";

    let description = [{
        An operation that casts the value to another qualified-type.
    }];
}

def Sin : Tiny_Op<"sin", [UnaryTrait, ElementwiseMappable]> {
    let summary = "Sine of the value.";

    let description = [{
        An operation that computes the sine of the value.
    }];
}

def Sqrt : Tiny_Op<"sqrt", [UnaryTrait, ElementwiseMappable]> {
    let summary = "Square-root of the value";

    let description = [{
        An operation that computes the square-root of the value.
    }];
}

def Neg : Tiny_Op<"neg", [UnaryTrait, ElementwiseMappable]> {
    let summary = "Negate the value.";

    let description = [{
        An operation that computes the negation of the value.
    }];
}

/* 
--------------------------------------------------- 
------------------- REDUCE OPS --------------------
--------------------------------------------------- */

/* 
--------------------------------------------------- 
------------------- BINARY OPS --------------------
--------------------------------------------------- */

/* 
--------------------------------------------------- 
-------------------- LOAD OPS ---------------------
--------------------------------------------------- */

/* 
--------------------------------------------------- 
------------------ TERNARY OPS --------------------
--------------------------------------------------- */


#endif // TINY_OPS