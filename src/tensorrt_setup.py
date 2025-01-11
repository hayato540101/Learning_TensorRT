import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import os


def make_trt_network():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    return builder, network

def make_engine(builder, network, calibration_data, input_shape, input_dtype):
    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    
    config.set_flag(trt.BuilderFlag.INT8)
    # キャリブレータを設定 (ここではキャリブレーションサンプルとして1サンプルのリストを渡す例)
    calibrator = MyCalibrator(calibration_data, input_shape, input_dtype)
    config.int8_calibrator = calibrator

    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    return engine, context

def execute_predict(context, input_data, output_shape, input_dtype, output_dtype, d_input, d_output, bindings, stream):
    cuda.memcpy_htod(d_input, input_data)
    context.execute_v2(bindings)
    output = np.empty(output_shape, dtype=output_dtype)
    cuda.memcpy_dtoh(output, d_output)
    return output


class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, input_shape, input_dtype):
        # 必ず親クラスのコンストラクタを呼ぶ
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.calibration_data = calibration_data  # 複数サンプルの場合はリストにしておく
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.cache_file = "calibration.cache"
        # 入力バッファを確保 (キャリブレーション時にGPUへ転送)
        self.device_input = cuda.mem_alloc(int(np.prod(input_shape)) * np.dtype(input_dtype).itemsize)
        self.current_index = 0

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        print("====== MyCalibrator > get_batch ======")
        if self.current_index < len(self.calibration_data):
            # 現在のバッチデータを取得
            batch = self.calibration_data[self.current_index]
            # 必要に応じ型変換・連続化
            batch = np.ascontiguousarray(batch.astype(self.input_dtype))
            cuda.memcpy_htod(self.device_input, batch)
            self.current_index += 1
            return [int(self.device_input)]
        else:
            # これ以上サンプルがなければ None を返す
            return None

    def read_calibration_cache(self):
        return None
        # if os.path.exists(self.cache_file):
        #     with open(self.cache_file, "rb") as f:
        #         return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)