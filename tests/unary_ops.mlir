// RUN: tiny-opt %s | FileCheck %s
func.func @simple_noop() -> () {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.noop %0 : tensor<3xf32> -> tensor<3xf32>

    func.return
}