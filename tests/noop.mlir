// RUN: tiny-opt %s --tiny-remove-redundant | FileCheck %s

tiny.func @simple_noop() -> tensor<3xf32> {
    // CHECK: tiny.constant
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    // CHECK-NOT: tiny.noop
    %1 = tiny.noop %0 : tensor<3xf32> -> tensor<3xf32>

    tiny.return %1 : tensor<3xf32>
}

tiny.func @cast_noop_erase() -> tensor<3xf32> {
    // CHECK: tiny.constant
    %0 = tiny.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    // CHECK-NOT: tiny.cast
    %1 = tiny.cast %0 : tensor<3xf32> -> tensor<3xf32>

    tiny.return %1 : tensor<3xf32>
}

