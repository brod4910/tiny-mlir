// RUN: tiny-opt %s | FileCheck %s


tiny.func @simple_empty() -> tensor<10x10xi8> {
    %s = tiny.shape[10,10] : !tiny.shape<10x10xi8>
    %0 = tiny.empty %s : tensor<10x10xi8>

    tiny.return %0 : tensor<10x10xi8>
}