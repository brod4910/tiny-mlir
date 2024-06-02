// RUN: tiny-opt %s | FileCheck %s

// CHECK-LABEL: @simple_where
tiny.func @simple_where(%arg0: tensor<10x1xi8>, %arg1: tensor<10x1xi8>, %arg2: tensor<10x1xi8>) -> tensor<10x1xi8> {
    // CHECK: tiny.where %arg0, %arg1, %arg2 : (tensor<10x1xi8>, tensor<10x1xi8>, tensor<10x1xi8>) -> tensor<10x1xi8>
    %0 = tiny.where %arg0, %arg1, %arg2 : (tensor<10x1xi8>, tensor<10x1xi8>, tensor<10x1xi8>) -> tensor<10x1xi8>

    tiny.return %0 : tensor<10x1xi8>
}

// CHECK-LABEL: @where_broadcast
tiny.func @where_broadcast(%arg0: tensor<10x10xi8>, %arg1: tensor<10x1xi8>, %arg2: tensor<1x10xi8>) -> tensor<10x10xi8> {
    // CHECK: tiny.where %arg0, %arg1, %arg2 : (tensor<10x10xi8>, tensor<10x1xi8>, tensor<1x10xi8>) -> tensor<10x10xi8>
    %0 = tiny.where %arg0, %arg1, %arg2 : (tensor<10x10xi8>, tensor<10x1xi8>, tensor<1x10xi8>) -> tensor<10x10xi8>

    tiny.return %0 : tensor<10x10xi8>
}

// CHECK-LABEL: @simple_mulacc
tiny.func @simple_mulacc(%arg0: tensor<10x1xi8>, %arg1: tensor<10x1xi8>, %arg2: tensor<10x1xi8>) -> tensor<10x1xi8> {
    // CHECK: tiny.mulacc %arg0, %arg1, %arg2 : (tensor<10x1xi8>, tensor<10x1xi8>, tensor<10x1xi8>) -> tensor<10x1xi8>
    %0 = tiny.mulacc %arg0, %arg1, %arg2 : (tensor<10x1xi8>, tensor<10x1xi8>, tensor<10x1xi8>) -> tensor<10x1xi8>

    tiny.return %0 : tensor<10x1xi8>
}

// CHECK-LABEL: @mulacc_broadcast
tiny.func @mulacc_broadcast(%arg0: tensor<10x1xi8>, %arg1: tensor<1x10xi8>, %arg2: tensor<10x1xi8>) -> tensor<10x10xi8> {
    // CHECK: tiny.mulacc %arg0, %arg1, %arg2 : (tensor<10x1xi8>, tensor<1x10xi8>, tensor<10x1xi8>) -> tensor<10x10xi8>
    %0 = tiny.mulacc %arg0, %arg1, %arg2 : (tensor<10x1xi8>, tensor<1x10xi8>, tensor<10x1xi8>) -> tensor<10x10xi8>

    tiny.return %0 : tensor<10x10xi8>
}