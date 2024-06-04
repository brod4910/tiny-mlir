// RUN: tiny-opt %s | FileCheck %s

tiny.func @simple_load(%arg0: tensor<10x1xi8>) -> tensor<1xi8> {
    %0 = tiny.load %arg0[0, 0] : (tensor<10x1xi8>) -> tensor<1xi8>

    tiny.return %0 : tensor<1xi8>
}