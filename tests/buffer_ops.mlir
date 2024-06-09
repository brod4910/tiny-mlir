// RUN: tiny-opt %s | FileCheck %s


tiny.func @simple_load(%arg0: tensor<10x1xi8>) -> tensor<1xi8> {
    %st = tiny.slice[1::] -> !tiny.slice<1::>
    %0 = tiny.load %arg0[%st] : (tensor<10x1xi8>, !tiny.slice<1::>) -> tensor<1xi8>

    tiny.return %0 : tensor<1xi8>
}