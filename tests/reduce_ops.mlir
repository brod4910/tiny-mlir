// RUN: tiny-opt %s | FileCheck %s

// CHECK-LABEL: @simple_max
tiny.func @simple_max(%arg0: tensor<10x1xi8>) -> tensor<1xi8> {
    %0 = tiny.max %arg0 : (tensor<10x1xi8>) -> tensor<1xi8>

    tiny.return %0 : tensor<1xi8>
}

// CHECK-LABEL: @simple_sum
tiny.func @simple_sum(%arg0: tensor<10x1xi8>, %axis: i32) -> tensor<10xi8> {
    %0 = tiny.sum %arg0, %axis : (tensor<10x1xi8>, i32) -> tensor<10xi8>

    tiny.return %0 : tensor<10xi8>
}
