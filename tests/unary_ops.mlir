// RUN: tiny-opt %s | FileCheck %s
tiny.func @simple_constant() -> tensor<3xf32> {
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>

    tiny.return %0 : tensor<3xf32>
}

// RUN: tiny-opt %s | FileCheck %s
tiny.func @simple_constant_sign() -> tensor<1xi8> {
    %0 = tiny.constant dense<[1]> : tensor<1xi8>

    tiny.return %0 : tensor<1xi8>
}

// RUN: tiny-opt %s | FileCheck %s
tiny.func @simple_exp2() -> tensor<3xf32> {
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.exp2 %0 : tensor<3xf32> -> tensor<3xf32>
    
    tiny.return %1 : tensor<3xf32>
}

// RUN: tiny-opt %s --tiny-erase-noop | FileCheck %s
tiny.func @simple_noop() -> tensor<3xf32> {
    // CHECK: tiny.constant
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    // CHECK-NOT: tiny.noop
    %1 = tiny.noop %0 : tensor<3xf32> -> tensor<3xf32>

    tiny.return %1 : tensor<3xf32>
}

// RUN: tiny-opt %s --tiny-erase-noop --tiny-cast-noop | FileCheck %s
tiny.func @cast_noop_erase() -> tensor<3xf32> {
    // CHECK: tiny.constant
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    // CHECK-NOT: tiny.cast
    %1 = tiny.cast %0 : tensor<3xf32> -> tensor<3xf32>

    tiny.return %1 : tensor<3xf32>
}

// RUN: tiny-opy %s | FileCheck %s
tiny.func @simple_cast() -> tensor<3xf16> {
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.cast %0 : tensor<3xf32> -> tensor<3xf16>

    tiny.return %1 : tensor<3xf16>
}
