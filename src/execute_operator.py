import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit 
from tensorrt_setup import make_trt_network, make_engine, execute_predict
from op.base import Operator
import json


def prepare_buffers(input_shape, output_shape, input_dtype, output_dtype):
    d_input = cuda.mem_alloc(int(np.prod(input_shape)) * np.dtype(input_dtype).itemsize)
    d_output = cuda.mem_alloc(int(np.prod(output_shape)) * np.dtype(output_dtype).itemsize)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    return d_input, d_output, bindings, stream

def print_data(input_data, output, input_shape, output_shape):
    print("input_data:\n", input_data)
    print("output:\n", output)
    print("input_shape:\n", input_shape)
    print("output_shape:\n", output_shape)

def dump_debug_json(engine):
    inspector = engine.create_engine_inspector()
    engine_info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    engine_info_json = json.loads(engine_info)
    with open("engine_info.json", "w") as f:
        json.dump(engine_info_json, f, indent=4)


if __name__ == "__main__":
    builder, network = make_trt_network()

    # operator
    operator_name = "conv"
    operator = Operator(operator_name, network)
    input_shape, output_shape, input_dtype, output_dtype, input_data = operator.create_network()
    calibration_data = [input_data]
    engine, context = make_engine(builder, network, calibration_data, input_shape, input_dtype)
    d_input, d_output, bindings, stream = prepare_buffers(input_shape, output_shape, input_dtype, output_dtype)
    output = execute_predict(context, input_data, output_shape, input_dtype, output_dtype, d_input, d_output, bindings, stream)

    print_data(input_data, output, input_shape, output_shape)
    dump_debug_json(engine)

"""
export PYTHONPATH=$PYTHONPATH:/workspace
python3 src/execute_operator.py
"""