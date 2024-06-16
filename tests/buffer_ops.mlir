// RUN: tiny-opt %s | FileCheck %s


tiny.func @simple_load(%arg0: tensor<10x1xi8>) -> tensor<1x1xi8> {
    %st = tiny.slice[0, 1] : !tiny.slice<0, 1>
    %0 = tiny.load %arg0[%st] : (tensor<10x1xi8>, !tiny.slice<0, 1>) -> tensor<1x1xi8>

    tiny.return %0 : tensor<1x1xi8>
}


tiny.func @simple_store(%arg0: tensor<10x1xi8>, %arg1: tensor<10x1xi8>) -> tensor<10x1xi8> {
    %slice1 = tiny.slice[1] : !tiny.slice<1>

    %slice5 = tiny.slice[5] : !tiny.slice<5>

    tiny.store %arg0[%slice1, %slice1], %arg1[%slice5, %slice1] : (tensor<10x1xi8>, !tiny.slice<1>, !tiny.slice<1>, tensor<10x1xi8>, !tiny.slice<5>, !tiny.slice<1>) -> ()

    tiny.return %arg1 : tensor<10x1xi8>
}