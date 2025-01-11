# conv.py

import numpy as np
import tensorrt as trt
from .registry import register_operator  # レジストリのデコレータをインポート

@register_operator("conv")
def create_conv_network(network):
    def set_input(network):
        input_data = np.array(
            [[[[ -3.0, -2.0, -1.0, -2.0, -1.0],
               [10.0, -25.0,  0.0, -2.0, -1.0],
               [ 1.0,   2.0, -2.0, -2.0, -1.0],
               [10.0, -25.0,  0.0, -2.0, -1.0],
               [ -3.0, -2.0, -1.0, -2.0, -1.0]]]],
            dtype=np.float32
        )
        input_dtype = input_data.dtype
        # 入力テンソル (バッチ, チャネル1, 高さ5, 幅5)
        batch = 1; channel = 1; height = 5; width = 5
        input_shape = (batch, channel, height, width)
        inp = network.add_input("input", trt.float32, input_shape)
        # inp = network.add_input("input", trt.int8, input_shape)
        return inp, input_dtype, input_shape, input_data

    def set_convolution(network, inp):
        num_filter = 3
        # カーネル (出力チャネル 3, 入力チャネル 1, 3x3)
        kernel_shape = (3, 3)
        w = np.array([
            [[0.3, -0.8, 1.0],
            [0.5, -0.5, 0.0],
            [0.4, -0.2, 0.9]],
            [[0.4, -0.7, 0.8],
            [0.3, -0.2, 1.0],
            [0.3,  0.2, 0.3]],
            [[0.1, -0.2, 0.3],
            [0.1, -0.2, 0.3],
            [0.1, -0.2, 0.9]],
        ], dtype=np.float32).reshape(num_filter, 1, kernel_shape[0], kernel_shape[1])
        weights = trt.Weights(w)

        conv = network.add_convolution_nd(input=inp, num_output_maps=num_filter, kernel_shape=kernel_shape, kernel=weights)
        return conv, num_filter

    def set_output(network, conv, num_filter):
    # def set_output(network, conv, num_filter, input_max, weights):
        output_tensor = conv.get_output(0)
        output_tensor.name = "output"
        
        # 出力テンソルの動的レンジを設定
        # weights_array = weights.numpy()  # Weightsオブジェクトからnumpy配列を取得
        # output_max = np.abs(weights_array).max() * input_max  # 概算の出力範囲
        # output_tensor.set_dynamic_range(-output_max, output_max)
        
        # 出力としてマークしてから型を設定
        network.mark_output(output_tensor)
        # output_tensor.dtype = trt.int8
        
        output_dtype = trt.nptype(output_tensor.dtype)
        output_shape = (1, num_filter, 3, 3)
        return output_shape, output_dtype

    inp, input_dtype, input_shape, input_data = set_input(network)
    conv, num_filter = set_convolution(network, inp)
    output_shape, output_dtype = set_output(network, conv, num_filter)

    print("\tinput_dtype:", input_dtype)
    print("\toutput_dtype:", output_dtype)
    return input_shape, output_shape, input_dtype, output_dtype, input_data

