import sys 
sys.path.append("TensorizeSet_app") 
from typing import Tuple, Union
from typing_extensions import Literal
import numpy as np

import tvm
from tvm import relay
from testing import extract_task

def fused_nn_conv2d(
    input_shape: Tuple[int, int, int, int],
    input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    weight_shape: Tuple[int, int, int, int],
    weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    channels: Tuple[int],
    kernel_size: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    groups = 1
):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    
    x = relay.nn.conv2d(data=input_data, weight=weight_data, strides=strides, padding=padding,
                        channels=channels, kernel_size=kernel_size, data_layout=input_layout,
                        kernel_layout=kernel_layout, out_dtype=out_dtype, groups=groups)
    
    relay_mod = tvm.IRModule.from_expr(x)
    
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    
    return relay_mod, params

def fused_nn_conv2d_template(
    input_shape: Tuple[int, int, int, int],
    weight_shape: Tuple[int, int, int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    channels: Tuple[int],
    kernel_size: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    groups = 1
):
    return locals()


def fused_nn_conv2d_add(
    input_shape: Tuple[int, int, int, int],
    input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    weight_shape: Tuple[int, int, int, int],
    weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    channels: Tuple[int],
    kernel_size: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    
    add_shape: Tuple[int, int, int, int],
    add_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    groups = 1
):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    
    x = relay.nn.conv2d(data=input_data, weight=weight_data, strides=strides, padding=padding,
                        channels=channels, kernel_size=kernel_size, data_layout=input_layout,
                        kernel_layout=kernel_layout, out_dtype=out_dtype, groups = groups)
    
    add_data = relay.var("add_data", shape=add_shape, dtype=add_dtype)
    
    x = relay.add(x, add_data)
    
    relay_mod = tvm.IRModule.from_expr(x)
    
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    
    return relay_mod, params
    
def fused_nn_conv2d_add_template(
    input_shape: Tuple[int, int, int, int],
    weight_shape: Tuple[int, int, int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    channels: Tuple[int],
    kernel_size: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    
    add_shape: Tuple[int, int, int, int],
    groups = 1
):
    return locals()

def fused_nn_conv2d_add_nn_relu(
    input_shape: Tuple[int, int, int, int],
    input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    weight_shape: Tuple[int, int, int, int],
    weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    channels: Tuple[int],
    kernel_size: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    
    add_shape: Tuple[int, int, int, int],
    add_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    
    groups = 1,
):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    
    x = relay.nn.conv2d(data=input_data, weight=weight_data, strides=strides, padding=padding,
                        channels=channels, kernel_size=kernel_size, data_layout=input_layout,
                        kernel_layout=kernel_layout, out_dtype=out_dtype, groups=groups)
    
    add_data = relay.var("add_data", shape=add_shape, dtype=add_dtype)
    
    x = relay.add(x, add_data)
    
    x = relay.nn.relu(x)
    
    relay_mod = tvm.IRModule.from_expr(x)
    
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    
    return relay_mod, params

def fused_nn_conv2d_add_nn_relu_template(
    input_shape: Tuple[int, int, int, int],
    weight_shape: Tuple[int, int, int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    channels: Tuple[int],
    kernel_size: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    
    add_shape: Tuple[int, int, int, int],
    
    groups = 1,
):
    return locals()

# rm, p = fused_nn_conv2d_add_nn_relu((1, 224, 224, 3), "float16", (7, 7, 3, 64), 'float16',
#                                     (2, 2), (3, 3), 64, (7, 7), 'NHWC', 'HWIO', 'float16',
#                                     (1, 1, 1, 64), 'float16')

# extract_task.test_extract_task(rm, 'nvidia/nvidia-a100', p)


def fused_nn_conv2d_add_add_nn_relu(
    input_shape: Tuple[int, int, int, int],
    input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    weight_shape: Tuple[int, int, int, int],
    weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    channels: Tuple[int],
    kernel_size: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    
    add1_shape: Tuple[int, int, int, int],
    add2_shape: Tuple[int, int, int, int],
    add_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    
    groups = 1,
):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    
    x = relay.nn.conv2d(data=input_data, weight=weight_data, strides=strides, padding=padding,
                        channels=channels, kernel_size=kernel_size, data_layout=input_layout,
                        kernel_layout=kernel_layout, out_dtype=out_dtype, groups=groups)
    
    add1_data = relay.var("add1_data", shape=add1_shape, dtype=add_dtype)
    
    x = relay.add(x, add1_data)
    
    add2_data = relay.var("add2_data", shape=add2_shape, dtype=add_dtype)
    
    x = relay.add(x, add2_data)
    
    x = relay.nn.relu(x)
    
    relay_mod = tvm.IRModule.from_expr(x)
    
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    
    return relay_mod, params

def fused_nn_conv2d_add_add_nn_relu_template(
    input_shape: Tuple[int, int, int, int],
    weight_shape: Tuple[int, int, int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    channels: Tuple[int],
    kernel_size: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    
    add1_shape: Tuple[int, int, int, int],
    add2_shape: Tuple[int, int, int, int],
    
    groups = 1,
):
    return locals()
# rm, p = fused_nn_conv2d_add_add_nn_relu((1, 56, 56, 64), "float16", (1, 1, 64, 256), 'float16',
#                                     (1, 1), (0, 0), 256, (1, 1), 'NHWC', 'HWIO', 'float16',
#                                     (1, 1, 1, 256), 'float16', (1, 56, 56, 256), 'float16')

# extract_task.test_extract_task(rm, 'nvidia/nvidia-a100', p)

def fused_nn_dense(
    input_shape: Tuple[int, int],
    input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    weight_shape: Tuple[int, int],
    weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    
    x = relay.nn.dense(input_data, weight_data, out_dtype=out_dtype)
    
    relay_mod = tvm.IRModule.from_expr(x)
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    
    return relay_mod, params

def fused_nn_dense_template(
    input_shape: Tuple[int, int],
    weight_shape: Tuple[int, int],
):
    return locals()

def fused_nn_dense_add(
    input_shape: Tuple[int, int],
    input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    weight_shape: Tuple[int, int],
    weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    
    add_shape: Tuple[int, int],
    add_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    
    x = relay.nn.dense(input_data, weight_data, out_dtype=out_dtype)
    add_data = relay.var("add_data", shape=add_shape, dtype=add_dtype)
    
    x = relay.add(x, add_data)
    
    relay_mod = tvm.IRModule.from_expr(x)
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    
    return relay_mod, params

def fused_nn_dense_add_template(
    input_shape: Tuple[int, int],
    weight_shape: Tuple[int, int],
    
    add_shape: Tuple[int, int],
):
    return locals()
# rm, p = fused_nn_dense_add((1, 512), 'float16', (1000, 512), 'float16', 'float16', 
#                            (1, 1000), 'float16')

# extract_task.test_extract_task(rm, 'nvidia/nvidia-a100', p)

def fused_nn_batch_matmul(
    input_shape: Tuple[int, int],
    input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    weight_shape: Tuple[int, int],
    weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    
    x = relay.nn.batch_matmul(input_data, weight_data, out_dtype=out_dtype, 
                              transpose_a=False,
                              transpose_b=True)
    
    relay_mod = tvm.IRModule.from_expr(x)
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    
    return relay_mod, params

def fused_nn_batch_matmul_template(
    input_shape: Tuple[int, int],
    weight_shape: Tuple[int, int],
):
    return locals()
# rm, p = fused_nn_batch_matmul((12, 128, 128), 'float16', (12, 128, 64), 'float16', 'float16')

# extract_task.test_extract_task(rm, 'nvidia/nvidia-a100', p)

def fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast(
    input_shape: Tuple[int, int, int, int],
    input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    weight_shape: Tuple[int, int, int, int],
    weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    channels: Tuple[int],
    kernel_size: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    
    add_shape: Tuple[int, int, int, int],
    add_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    
    groups = 1,
):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    
    x = relay.nn.conv2d(data=input_data, weight=weight_data, strides=strides, padding=padding,
                        channels=channels, kernel_size=kernel_size, data_layout=input_layout,
                        kernel_layout=kernel_layout, out_dtype=out_dtype, groups=groups)
    
    add_data = relay.var("add_data", shape=add_shape, dtype=add_dtype)
    
    x = relay.add(x, add_data)
    
    x = relay.nn.relu(x)

    x = relay.add(x, relay.const(256, dtype='int32'))
    x = relay.right_shift(x, relay.const(9, dtype='int32'))
    x = relay.clip(x, a_min=-127., a_max=127.)
    x = relay.cast(x, "uint8")
    
    relay_mod = tvm.IRModule.from_expr(x)
    
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    
    return relay_mod, params

# rm, p = fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast(
#     (1, 56, 56, 64), "uint8", (3, 3, 64, 64), "int8", (1, 1), (1, 1), 64, (3, 3), "NHWC", "HWIO",
#     "int32", (1, 56, 56, 64), "int32" 
# )

# extract_task.test_extract_task(rm, "llvm -mcpu=cascadelake -num-cores 4", p)


def fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast_template(
    input_shape: Tuple[int, int, int, int],
    weight_shape: Tuple[int, int, int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    channels: Tuple[int],
    kernel_size: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    
    add_shape: Tuple[int, int, int, int],
    
    groups = 1,
):
    return locals()

def fused_nn_conv2d_add_right_shift_clip_cast(
    input_shape: Tuple[int, int, int, int],
    input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    weight_shape: Tuple[int, int, int, int],
    weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    channels: Tuple[int],
    kernel_size: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    
    add_shape: Tuple[int, int, int, int],
    add_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    
    groups = 1,
):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    
    x = relay.nn.conv2d(data=input_data, weight=weight_data, strides=strides, padding=padding,
                        channels=channels, kernel_size=kernel_size, data_layout=input_layout,
                        kernel_layout=kernel_layout, out_dtype=out_dtype, groups=groups)
    
    add_data = relay.var("add_data", shape=add_shape, dtype=add_dtype)
    
    x = relay.add(x, add_data)
    
    x = relay.right_shift(x, relay.const(8, dtype='int32'))
    x = relay.clip(x, a_min=-127, a_max=127)
    x = relay.cast(x, "uint8")
    
    relay_mod = tvm.IRModule.from_expr(x)
    
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    
    return relay_mod, params

# rm, p = fused_nn_conv2d_add_right_shift_clip_cast(
#     (1, 56, 56, 64), "uint8", (1, 1, 64, 256), "int8", (1, 1), (0, 0), 256, (1, 1), "NHWC", "HWIO",
#     "int32", (1, 1, 1, 256), "int32" 
# )

# extract_task.test_extract_task(rm, "llvm -mcpu=cascadelake -num-cores 4", p)

def fused_nn_conv2d_add_right_shift_clip_cast_template(
    input_shape: Tuple[int, int, int, int],
    weight_shape: Tuple[int, int, int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    channels: Tuple[int],
    kernel_size: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    
    add_shape: Tuple[int, int, int, int],
    
    groups = 1,
):
    return locals()


def fused_nn_conv3d(
    input_shape: Tuple[int, int, int, int],
    input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    weight_shape: Tuple[int, int, int, int],
    weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    
    x = relay.nn.conv3d(data=input_data, 
                        weight=weight_data, 
                        strides=strides, 
                        padding=padding, 
                        data_layout=input_layout, 
                        kernel_layout=kernel_layout,
                        out_dtype=out_dtype)
    
    relay_mod = tvm.IRModule.from_expr(x)
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    return relay_mod, params

def fused_nn_conv3d_template(
    input_shape: Tuple[int, int, int, int],
    weight_shape: Tuple[int, int, int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    input_layout,
    kernel_layout, 
):
    return locals()


def fused_nn_conv3d_add(
    input_shape: Tuple[int, int, int, int],
    input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    weight_shape: Tuple[int, int, int, int],
    weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    
    add_shape: Tuple[int, int, int, int],
    add_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    
    x = relay.nn.conv3d(data=input_data, 
                        weight=weight_data, 
                        strides=strides, 
                        padding=padding, 
                        data_layout=input_layout, 
                        kernel_layout=kernel_layout,
                        out_dtype=out_dtype)
    
    add_data = relay.var("add_data", shape=add_shape, dtype=add_dtype)
    x = relay.add(x, add_data)
    
    relay_mod = tvm.IRModule.from_expr(x)
    
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    
    return relay_mod, params
    
def fused_nn_conv3d_add_template(
    input_shape: Tuple[int, int, int, int],
    weight_shape: Tuple[int, int, int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    input_layout,
    kernel_layout,
    
    add_shape: Tuple[int, int, int, int],
):
    return locals()

def fused_nn_conv3d_add_nn_relu(
    input_shape: Tuple[int, int, int, int],
    input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    weight_shape: Tuple[int, int, int, int],
    weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
    
    add_shape: Tuple[int, int, int, int],
    add_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    
    x = relay.nn.conv3d(data=input_data, 
                        weight=weight_data, 
                        strides=strides, 
                        padding=padding, 
                        data_layout=input_layout, 
                        kernel_layout=kernel_layout,
                        out_dtype=out_dtype)
    add_data = relay.var("add_data", shape=add_shape, dtype=add_dtype)
    x = relay.add(x, add_data)
    x = relay.nn.relu(x)
    
    relay_mod = tvm.IRModule.from_expr(x)
    
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    
    return relay_mod, params

def fused_nn_conv3d_add_nn_relu_template(
    input_shape: Tuple[int, int, int, int],
    weight_shape: Tuple[int, int, int, int],
    strides: Tuple[int, int],
    padding: Tuple[int, int],
    input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
    kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
    add_shape: Tuple[int, int, int, int],
):
    return locals()


SUPPORT = {"fused_nn_conv2d": fused_nn_conv2d,
           "fused_nn_conv2d_add": fused_nn_conv2d_add,
           "fused_nn_conv2d_add_nn_relu": fused_nn_conv2d_add_nn_relu,
           "fused_nn_conv2d_add_add_nn_relu": fused_nn_conv2d_add_add_nn_relu,
           "fused_nn_dense": fused_nn_dense,
           "fused_nn_dense_add": fused_nn_dense_add,
           "fused_nn_batch_matmul": fused_nn_batch_matmul,
           "fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast": fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast,
           "fused_nn_conv2d_add_right_shift_clip_cast": fused_nn_conv2d_add_right_shift_clip_cast,
           "fused_nn_conv3d":fused_nn_conv3d,
           "fused_nn_conv3d_add":fused_nn_conv3d_add,
           "fused_nn_conv3d_add_nn_relu":fused_nn_conv3d_add_nn_relu
           }
# para_dict Template

SUPPORT_TEMPLATE = {"fused_nn_conv2d_template": fused_nn_conv2d_template,
           "fused_nn_conv2d_add_template": fused_nn_conv2d_add_template,
           "fused_nn_conv2d_add_nn_relu_template": fused_nn_conv2d_add_nn_relu_template,
           "fused_nn_conv2d_add_add_nn_relu_template": fused_nn_conv2d_add_add_nn_relu_template,
           "fused_nn_dense_template": fused_nn_dense_template,
           "fused_nn_dense_add_template": fused_nn_dense_add_template,
           "fused_nn_batch_matmul_template": fused_nn_batch_matmul_template,
           "fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast_template": fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast_template,
           "fused_nn_conv2d_add_right_shift_clip_cast_template": fused_nn_conv2d_add_right_shift_clip_cast_template,
           "fused_nn_conv3d_template": fused_nn_conv3d_template,
           "fused_nn_conv3d_add_template": fused_nn_conv3d_add_template,
           "fused_nn_conv3d_add_nn_relu_template": fused_nn_conv3d_add_nn_relu_template,
           }

class relayops:
    def __init__(self,
                 intput_dtype: str,
                 weight_dtype: str,
                 out_dtype: str,
                 ) -> None:
        self.input_dtype = intput_dtype
        self.weight_dtype = weight_dtype
        self.out_dtype = out_dtype
        self.add_dtype = out_dtype
        
    def get_mod_params(self, 
                       task_name: Union[Literal['fused_nn_conv2d'], Literal['fused_nn_conv2d_add'],
                                        Literal['fused_nn_conv2d_add_nn_relu'],
                                        Literal['fused_nn_conv2d_add_add_nn_relu'],
                                        Literal['fused_nn_dense'], Literal['fused_nn_dense_add'],
                                        Literal['fused_nn_batch_matmul'],
                                        Literal['fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast'],
                                        Literal['fused_nn_conv2d_add_right_shift_clip_cast']],
                       para_dict,
                       
        ):
        para_dict["input_dtype"] = self.input_dtype
        para_dict["weight_dtype"] = self.weight_dtype
        para_dict["out_dtype"] = self.out_dtype
        if task_name != 'fused_nn_dense' and task_name != 'fused_nn_conv2d' and\
            task_name != 'fused_nn_batch_matmul' and task_name != 'fused_nn_conv3d':
            para_dict["add_dtype"] = self.add_dtype
        f_get = SUPPORT[task_name]
        rm, p = f_get(**para_dict)
        return (rm, p) 
