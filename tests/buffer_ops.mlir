// RUN: tiny-opt %s | FileCheck %s


tiny.func @simple_load(%arg0: tensor<10x1xi8>) -> tensor<1xi8> {
    %c0 = arith.constant 1 : i32
    %st = tiny.slice[%c0, %c0] : (i32, i32) -> !tiny.slice<1, 1>
    %0 = tiny.load %arg0[%st] : (tensor<10x1xi8>, !tiny.slice<1, 1>) -> tensor<1xi8>

    tiny.return %0 : tensor<1xi8>
}