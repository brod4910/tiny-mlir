// RUN: tiny-opt %s | FileCheck %s

// CHECK-LABEL: @simple_max
tiny.func @simple_max(%arg0: tensor<10x1xi8>) -> tensor<1xi8> {
    // CHECK: tiny.max %arg0 : (tensor<10x1xi8>) -> tensor<1xi8>
    %0 = tiny.max %arg0 : (tensor<10x1xi8>) -> tensor<1xi8>

    tiny.return %0 : tensor<1xi8>
}


// CHECK-LABEL: @max
tiny.func @max(%arg0: tensor<10x2x5xi8>) -> tensor<5xi8> {
    // CHECK: tiny.max %arg0 {axis = -2 : i32} : (tensor<10x2x5xi8>) -> tensor<10x5xi8>
    %0 = tiny.max %arg0 {axis = -2 : i32} : (tensor<10x2x5xi8>) -> tensor<10x5xi8>
    // CHECK: tiny.max %0 {axis = -2 : i32} : (tensor<10x5xi8>) -> tensor<5xi8>
    %1 = tiny.max %0 {axis = -2 : i32} : (tensor<10x5xi8>) -> tensor<5xi8>

    tiny.return %1 : tensor<5xi8>
}

// CHECK-LABEL: @simple_sum
tiny.func @simple_sum(%arg0: tensor<10x1xi8>) -> tensor<10xi8> {
    // CHECK: tiny.sum %arg0 {axis = -1 : i32} : (tensor<10x1xi8>) -> tensor<10xi8>
    %0 = tiny.sum %arg0 {axis = -1 : i32} : (tensor<10x1xi8>) -> tensor<10xi8>

    tiny.return %0 : tensor<10xi8>
}

// CHECK-LABEL: @sum
tiny.func @sum(%arg0: tensor<100x50x25x5xi8>) -> tensor<100xi8> {
    // CHECK: tiny.sum %arg0 {axis = -1 : i32} : (tensor<100x50x25x5xi8>) -> tensor<100x50x25xi8>
    %0 = tiny.sum %arg0 {axis = -1 : i32} : (tensor<100x50x25x5xi8>) -> tensor<100x50x25xi8>
    // CHECK: tiny.sum %0 {axis = 1 : i32} : (tensor<100x50x25xi8>) -> tensor<100x25xi8>
    %1 = tiny.sum %0 {axis = 1 : i32} : (tensor<100x50x25xi8>) -> tensor<100x25xi8>
    // CHECK: tiny.sum %1 {axis = -1 : i32} : (tensor<100x25xi8>) -> tensor<100xi8>
    %2 = tiny.sum %1 {axis = -1 : i32} : (tensor<100x25xi8>) -> tensor<100xi8>

    tiny.return %2 : tensor<100xi8>
}