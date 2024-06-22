// RUN: tiny-opt %s | FileCheck %s

// CHECK-LABEL: @simple_bool_constant
tiny.func @simple_bool_constant() -> tensor<i1> {
    %0 = tiny.constant dense<true> : tensor<i1>

    tiny.return %0 : tensor<i1>
}

// CHECK-LABEL: @simple_tensor_constant
tiny.func @simple_tensor_constant() -> tensor<3xf32> {
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>

    tiny.return %0 : tensor<3xf32>
}

// CHECK-LABEL: @simple_constant_sign
tiny.func @simple_constant_sign() -> tensor<1xi8> {
    %0 = tiny.constant dense<[1]> : tensor<1xi8>

    tiny.return %0 : tensor<1xi8>
}

// CHECK-LABEL: @simple_noop
tiny.func @simple_noop() -> tensor<3xf32> {
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.noop %0 : tensor<3xf32> -> tensor<3xf32>

    tiny.return %1 : tensor<3xf32>
}

// CHECK-LABEL: @simple_exp2
tiny.func @simple_exp2() -> tensor<3xf32> {
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.exp2 %0 : tensor<3xf32> -> tensor<3xf32>
    
    tiny.return %1 : tensor<3xf32>
}

// CHECK-LABEL: @simple_log2
tiny.func @simple_log2() -> tensor<3xf16> {
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.log2 %0 : tensor<3xf32> -> tensor<3xf16>

    tiny.return %1 : tensor<3xf16>
}

// CHECK-LABEL: @simple_cast
tiny.func @simple_cast() -> tensor<3xf16> {
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.cast %0 : tensor<3xf32> -> tensor<3xf16>

    tiny.return %1 : tensor<3xf16>
}

// CHECK-LABEL: @simple_sin
tiny.func @simple_sin() -> tensor<3xf16> {
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.sin %0 : tensor<3xf32> -> tensor<3xf16>

    tiny.return %1 : tensor<3xf16>
}

// CHECK-LABEL: @simple_sqrt
tiny.func @simple_sqrt() -> tensor<3xf16> {
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.sqrt %0 : tensor<3xf32> -> tensor<3xf16>

    tiny.return %1 : tensor<3xf16>
}

// CHECK-LABEL: @simple_neg
tiny.func @simple_neg() -> tensor<3xf16> {
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.neg %0 : tensor<3xf32> -> tensor<3xf16>

    tiny.return %1 : tensor<3xf16>
}

// CHECK-LABEL: @simple_recip
tiny.func @simple_recip() -> tensor<3xf16> {
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.recip %0 : tensor<3xf32> -> tensor<3xf16>

    tiny.return %1 : tensor<3xf16>
}