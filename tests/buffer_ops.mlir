// RUN: tiny-opt %s | FileCheck %s


// tiny.func @simple_load(%arg0: tensor<10x1xi8>) -> tensor<1xi8> {
//     %c1 = arith.constant 1 : i32
//     %st = tiny.slice[%c1, %c1, %c1] : (i32, i32, i32) -> !tiny.slice<1, 1, 1>
//     %0 = tiny.load %arg0[%st] : (tensor<10x1xi8>, !tiny.slice<1, 1, 1>) -> tensor<1xi8>

//     tiny.return %0 : tensor<1xi8>
// }


tiny.func @simple_store(%arg0: tensor<10x1xi8>, %arg1: tensor<10x1xi8>) -> tensor<10x1xi8> {
    %c1 = arith.constant 1 : i32
    %slice1 = tiny.slice[%c1] : (i32) -> !tiny.slice<1>

    %c0 = arith.constant 1 : i32
    %slice0 = tiny.slice[%c1] : (i32) -> !tiny.slice<1>

    %c5 = arith.constant 5 : i32
    %slice5 = tiny.slice[%c5] : (i32) -> !tiny.slice<5>

    tiny.store %arg0[%slice1, %slice0], %arg1[%slice5, %slice1] : (tensor<10x1xi8>, !tiny.slice<1>, !tiny.slice<1>, tensor<10x1xi8>, !tiny.slice<5>, !tiny.slice<1>) -> ()

    tiny.return %arg1 : tensor<10x1xi8>
}