import sys 
sys.path.append("TensorizeSet_app") 
from typing import Any, Union, Tuple

import tvm
from tvm import relay
from typing_extensions import Literal
from relay_ops import (fused_nn_conv2d_template,
                       fused_nn_conv2d_add_template,
                       fused_nn_conv2d_add_nn_relu_template,
                       fused_nn_conv2d_add_add_nn_relu_template,
                       fused_nn_dense_template,
                       fused_nn_dense_add_template,
                       fused_nn_batch_matmul_template,
                       fused_nn_conv2d_add_right_shift_clip_cast_template,
                       fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast_template,
                       fused_nn_conv3d_template,
                       fused_nn_conv3d_add_template,
                       fused_nn_conv3d_add_nn_relu_template,
                       relayops,
                       )
import common

class net:
    def __init__(self,
                 name: str,
                 model_input: Tuple,
                 model_layout: Union[Literal["NHWC"], Literal["NCHW"], None],
                 model_dtype: Union[Literal["float16"], Literal["int8"], Literal["int8int8"],None], 
                 target,) -> None:
        self.name = name
        self.model_input = model_input
        self.model_layout = model_layout
        self.model_dtype = model_dtype
        self.target = target
        
        target_name = common.get_target_from_Target(self.target)
        
        self.model_name =  '-'.join([self.name, str(self.model_layout),
                                    ",".join(map(str, self.model_input)),
                                    target_name,
                                    str(self.model_dtype)])
        input_dtype = "float16"
        weight_dtype = "float16"
        out_dype = "float16"
        if target_name == "avx512" or target_name == 'vnni':
            input_dtype = "uint8"
            weight_dtype = "int8"
            out_dype = "int32"
        if target_name == "neon":
            input_dtype = "int8"
            weight_dtype = "int8"
            out_dype = "int32"
        self.op_utils = relayops(input_dtype, weight_dtype, out_dype)
        self.mod_params_list = []
        
    def collect(self):
        def pattern1(b, h, w, i, weight = 1):
            param_dict = fused_nn_conv2d_add_nn_relu_template(
                input_shape=(b, i, h, w),
                weight_shape=(128, i, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=128,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
                add_shape=(b, 128, 1, 1)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d_add_nn_relu',
                param_dict
            )
            self.mod_params_list.append((rm, p, weight))
        
        def pattern_bert(b, m, n, k, weight = 1):
            param_dict = fused_nn_batch_matmul_template(
                input_shape=(b, m, k),
                weight_shape=(b, n, k)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_batch_matmul',
                param_dict
            )
            self.mod_params_list.append((rm, p, weight))
            
        def pattern_resnet_add(batch, kh, kw, ic, oc, ih, iw, strides, padding, weight=1):
            param_dict = fused_nn_conv2d_add_template(
                input_shape=(batch, ic, ih, iw),
                weight_shape=(oc, ic, kh, kw),
                strides=(strides, strides),
                padding=(padding, padding),
                channels=oc,
                kernel_size=(kh, kw),
                input_layout="NCHW",
                kernel_layout="OIHW",
                add_shape=(batch, oc, 1, 1)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d_add',
                param_dict
            )
            self.mod_params_list.append((rm, p, weight))
        
        def pattern_resnet_add_relu(batch, kh, kw, ic, oc, ih, iw, strides, padding, weight=1):
            param_dict = fused_nn_conv2d_add_nn_relu_template(
                input_shape=(batch, ic, ih, iw),
                weight_shape=(oc, ic, kh, kw),
                strides=(strides, strides),
                padding=(padding, padding),
                channels=oc,
                kernel_size=(kh, kw),
                input_layout="NCHW",
                kernel_layout="OIHW",
                add_shape=(batch, oc, 1, 1)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d_add_nn_relu',
                param_dict
            )
            self.mod_params_list.append((rm, p, weight))
        
        def pattern_resnet_add_add_relu(batch, kh, kw, ic, oc, ih, iw, strides, padding, weight=1):
            param_dict = fused_nn_conv2d_add_add_nn_relu_template(
                input_shape=(batch, ic, ih, iw),
                weight_shape=(oc, ic, kh, kw),
                strides=(strides, strides),
                padding=(padding, padding),
                channels=oc,
                kernel_size=(kh, kw),
                input_layout="NCHW",
                kernel_layout="OIHW",
                add1_shape=(batch, oc, 1, 1),
                add2_shape=(batch, oc, 1, 1)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d_add_add_nn_relu',
                param_dict
            )
            self.mod_params_list.append((rm, p, weight))
        
        def pattern_inception_v3_add_relu(batch, kh, kw, ic, oc, ih, iw, strides, padding, weight = 1):
            param_dict = fused_nn_conv2d_add_nn_relu_template(
                input_shape=(batch, ic, ih, iw),
                weight_shape=(oc, ic, kh, kw),
                strides=(strides, strides),
                padding=padding,
                channels=oc,
                kernel_size=(kh, kw),
                input_layout="NCHW",
                kernel_layout="OIHW",
                add_shape=(batch, oc, 1, 1)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d_add_nn_relu',
                param_dict
            )
            self.mod_params_list.append((rm, p, weight))
        
        def pattern_mobilenet_add(batch, kh, kw, ic, oc, ih, iw, strides, padding, groups, weight = 1):
            param_dict = fused_nn_conv2d_add_template(
                input_shape=(batch, ic, ih, iw),
                weight_shape=(oc, ic, kh, kw),
                strides=(strides, strides),
                padding=(padding, padding),
                channels=oc,
                kernel_size=(kh, kw),
                input_layout="NCHW",
                kernel_layout="OIHW",
                add_shape=(batch, oc, 1, 1),
                groups=groups,
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d_add',
                param_dict
            )
            self.mod_params_list.append((rm, p, weight))
        
        def pattern_mobilenet_add_group(batch, kh, kw, oc, ic, ih, iw, strides, padding, groups, weight = 1):
            param_dict = fused_nn_conv2d_add_template(
                input_shape=(batch, ic * groups, ih, iw),
                weight_shape=(oc, ic, kh, kw),
                strides=(strides, strides),
                padding=(padding, padding),
                channels=oc,
                kernel_size=(kh, kw),
                input_layout="NCHW",
                kernel_layout="OIHW",
                add_shape=(batch, oc, 1, 1),
                groups=groups,
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d_add',
                param_dict
            )
            self.mod_params_list.append((rm, p, weight))
        def pattern_conv3d(N, D, H, W, I, O, kD, kH, kW, strides, padding, weight, is_add, is_relu):
            input_shape = (N, D, H, W, I)
            weight_shape = (kD, kH, kW, I, O)
            if is_add:
                param_dict = fused_nn_conv3d_add_template(
                input_shape=input_shape,
                weight_shape=weight_shape,
                strides=strides,
                padding=padding,
                input_layout="NDHWC",
                kernel_layout="DHWIO",
                add_shape=(N, 1, 1, 1, 1)
                )
                (rm, p) = self.op_utils.get_mod_params(
                    'fused_nn_conv3d_add',
                    param_dict
                )
            elif is_add and is_relu:
                param_dict = fused_nn_conv3d_add_nn_relu_template(
                input_shape=input_shape,
                weight_shape=weight_shape,
                strides=strides,
                padding=padding,
                input_layout="NDHWC",
                kernel_layout="DHWIO",
                add_shape=(N, 1, 1, 1, 1)
                )
                (rm, p) = self.op_utils.get_mod_params(
                    'fused_nn_conv3d_add_nn_relu',
                    param_dict
                )
            else:
                param_dict = fused_nn_conv3d_template(
                    input_shape=input_shape,
                    weight_shape=weight_shape,
                    strides=strides,
                    padding=padding,
                    input_layout="NDHWC",
                    kernel_layout="DHWIO",
                )
                (rm, p) = self.op_utils.get_mod_params(
                    'fused_nn_conv3d',
                    param_dict
                )
            self.mod_params_list.append((rm, p, weight))
        
        # densenet121
        if self.name == "densenet_121" and self.model_input == (1, 3, 224, 224) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern1(1, 7, 7, 960, 1)
            pattern1(1, 7, 7, 928, 1)
            pattern1(1, 7, 7, 896, 1)
            pattern1(1, 7, 7, 864, 1)
            pattern1(1, 7, 7, 832, 1)
            pattern1(1, 7, 7, 800, 1)
            pattern1(1, 7, 7, 768, 1)
            pattern1(1, 7, 7, 736, 1)
            pattern1(1, 7, 7, 704, 1)
            pattern1(1, 7, 7, 672, 1)
            pattern1(1, 7, 7, 640, 1)
            pattern1(1, 7, 7, 608, 1)
            pattern1(1, 7, 7, 576, 1)
            pattern1(1, 7, 7, 544, 1)
            pattern1(1, 7, 7, 512, 1)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 128, 7, 7),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 16))
            
            
            pattern1(1, 14, 14, 992, 1)
            pattern1(1, 14, 14, 960, 1)
            pattern1(1, 14, 14, 928, 1)
            pattern1(1, 14, 14, 896, 1)
            pattern1(1, 14, 14, 864, 1)
            pattern1(1, 14, 14, 832, 1)
            pattern1(1, 14, 14, 800, 1)
            pattern1(1, 14, 14, 768, 1)
            pattern1(1, 14, 14, 736, 1)
            pattern1(1, 14, 14, 704, 1)
            pattern1(1, 14, 14, 672, 1)
            pattern1(1, 14, 14, 640, 1)
            pattern1(1, 14, 14, 608, 1)
            pattern1(1, 14, 14, 576, 1)
            pattern1(1, 14, 14, 544, 1)
            pattern1(1, 14, 14, 512, 1)
            pattern1(1, 14, 14, 480, 1)
            pattern1(1, 14, 14, 448, 1)
            pattern1(1, 14, 14, 416, 1)
            pattern1(1, 14, 14, 384, 1)
            pattern1(1, 14, 14, 352, 1)
            pattern1(1, 14, 14, 320, 1)
            pattern1(1, 14, 14, 288, 1)
            pattern1(1, 14, 14, 256, 1)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 128, 14, 14),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 16))
            
            
            pattern1(1, 28, 28, 480, 1)
            pattern1(1, 28, 28, 448, 1)
            pattern1(1, 28, 28, 384, 1)
            pattern1(1, 28, 28, 352, 1)
            pattern1(1, 28, 28, 288, 1)
            pattern1(1, 28, 28, 256, 1)
            pattern1(1, 28, 28, 224, 1)
            pattern1(1, 28, 28, 192, 1)
            pattern1(1, 28, 28, 160, 1)
            pattern1(1, 28, 28, 128, 1)
            pattern1(1, 28, 28, 480, 1)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 128, 28, 28),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            pattern1(1, 56, 56, 224, 1)
            pattern1(1, 56, 56, 192, 1)
            pattern1(1, 56, 56, 160, 1)
            pattern1(1, 56, 56, 128, 1)
            pattern1(1, 56, 56, 96)
            pattern1(1, 56, 56, 64)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 128, 56, 56),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 6))
            
            pattern1(1, 56, 56, 96, 1)
            pattern1(1, 56, 56, 64, 1)
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 3, 224, 224),
                weight_shape=(64, 3, 7, 7),
                strides=(2, 2),
                padding=(3, 3),
                channels=64,
                kernel_size=(7, 7),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d',
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 256, 56, 56),
                weight_shape=(128, 256, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=128,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 512, 28, 28),
                weight_shape=(256, 512, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=256,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 1024, 14, 14),
                weight_shape=(512, 1024, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=512,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 1024),
                weight_shape=(1000, 1024),
                add_shape=(1, 1000),
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "densenet_121" and self.model_input == (8, 3, 224, 224) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern1(8, 7, 7, 960)
            pattern1(8, 7, 7, 928)
            pattern1(8, 7, 7, 896)
            pattern1(8, 7, 7, 864)
            pattern1(8, 7, 7, 832)
            pattern1(8, 7, 7, 800)
            pattern1(8, 7, 7, 768)
            pattern1(8, 7, 7, 736)
            pattern1(8, 7, 7, 704)
            pattern1(8, 7, 7, 672)
            pattern1(8, 7, 7, 640)
            pattern1(8, 7, 7, 608)
            pattern1(8, 7, 7, 576)
            pattern1(8, 7, 7, 544)
            pattern1(8, 7, 7, 512)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 128, 7, 7),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 16))
            
            pattern1(8, 14, 14, 992)
            pattern1(8, 14, 14, 960)
            pattern1(8, 14, 14, 928)
            pattern1(8, 14, 14, 896)
            pattern1(8, 14, 14, 864)
            pattern1(8, 14, 14, 832)
            pattern1(8, 14, 14, 800)
            pattern1(8, 14, 14, 768)
            pattern1(8, 14, 14, 736)
            pattern1(8, 14, 14, 704)
            pattern1(8, 14, 14, 672)
            pattern1(8, 14, 14, 640)
            pattern1(8, 14, 14, 608)
            pattern1(8, 14, 14, 576)
            pattern1(8, 14, 14, 544)
            pattern1(8, 14, 14, 512)
            pattern1(8, 14, 14, 480)
            pattern1(8, 14, 14, 448)
            pattern1(8, 14, 14, 416)
            pattern1(8, 14, 14, 384)
            pattern1(8, 14, 14, 352)
            pattern1(8, 14, 14, 320)
            pattern1(8, 14, 14, 288)
            pattern1(8, 14, 14, 256)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 128, 14, 14),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 16))
            
            pattern1(8, 28, 28, 480)
            pattern1(8, 28, 28, 448)
            pattern1(8, 28, 28, 384)
            pattern1(8, 28, 28, 352)
            pattern1(8, 28, 28, 288)
            pattern1(8, 28, 28, 256)
            pattern1(8, 28, 28, 224)
            pattern1(8, 28, 28, 192)
            pattern1(8, 28, 28, 160)
            pattern1(8, 28, 28, 128)
            pattern1(8, 28, 28, 480)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 128, 28, 28),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            pattern1(8, 56, 56, 224)
            pattern1(8, 56, 56, 192)
            pattern1(8, 56, 56, 160)
            pattern1(8, 56, 56, 128)
            pattern1(8, 56, 56, 96)
            pattern1(8, 56, 56, 64)
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 3, 224, 224),
                weight_shape=(64, 3, 7, 7),
                strides=(2, 2),
                padding=(3, 3, 3, 3),
                channels=64,
                kernel_size=(7, 7),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 256, 56, 56),
                weight_shape=(128, 256, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=128,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 512, 28, 28),
                weight_shape=(256, 512, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=256,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 1024, 14, 14),
                weight_shape=(512, 1024, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=512,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_dense_template(
                input_shape=(8, 1024),
                weight_shape=(1024, 1024),
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "densenet_121" and self.model_input == (16, 3, 224, 224) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern1(16, 7, 7, 960)
            pattern1(16, 7, 7, 928)
            pattern1(16, 7, 7, 896)
            pattern1(16, 7, 7, 864)
            pattern1(16, 7, 7, 832)
            pattern1(16, 7, 7, 800)
            pattern1(16, 7, 7, 768)
            pattern1(16, 7, 7, 736)
            pattern1(16, 7, 7, 704)
            pattern1(16, 7, 7, 672)
            pattern1(16, 7, 7, 640)
            pattern1(16, 7, 7, 608)
            pattern1(16, 7, 7, 576)
            pattern1(16, 7, 7, 544)
            pattern1(16, 7, 7, 512)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 128, 7, 7),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 16))
            
            pattern1(16, 14, 14, 992)
            pattern1(16, 14, 14, 960)
            pattern1(16, 14, 14, 928)
            pattern1(16, 14, 14, 896)
            pattern1(16, 14, 14, 864)
            pattern1(16, 14, 14, 832)
            pattern1(16, 14, 14, 800)
            pattern1(16, 14, 14, 768)
            pattern1(16, 14, 14, 736)
            pattern1(16, 14, 14, 704)
            pattern1(16, 14, 14, 672)
            pattern1(16, 14, 14, 640)
            pattern1(16, 14, 14, 608)
            pattern1(16, 14, 14, 576)
            pattern1(16, 14, 14, 544)
            pattern1(16, 14, 14, 512)
            pattern1(16, 14, 14, 480)
            pattern1(16, 14, 14, 448)
            pattern1(16, 14, 14, 416)
            pattern1(16, 14, 14, 384)
            pattern1(16, 14, 14, 352)
            pattern1(16, 14, 14, 320)
            pattern1(16, 14, 14, 288)
            pattern1(16, 14, 14, 256)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 128, 14, 14),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 16))
            
            pattern1(16, 28, 28, 480)
            pattern1(16, 28, 28, 448)
            pattern1(16, 28, 28, 384)
            pattern1(16, 28, 28, 352)
            pattern1(16, 28, 28, 288)
            pattern1(16, 28, 28, 256)
            pattern1(16, 28, 28, 224)
            pattern1(16, 28, 28, 192)
            pattern1(16, 28, 28, 160)
            pattern1(16, 28, 28, 128)
            pattern1(16, 28, 28, 480)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 128, 28, 28),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            pattern1(16, 56, 56, 224)
            pattern1(16, 56, 56, 192)
            pattern1(16, 56, 56, 160)
            pattern1(16, 56, 56, 128)
            pattern1(16, 56, 56, 96)
            pattern1(16, 56, 56, 64)
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 3, 224, 224),
                weight_shape=(64, 3, 7, 7),
                strides=(2, 2),
                padding=(3, 3),
                channels=64,
                kernel_size=(7, 7),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 256, 56, 56),
                weight_shape=(128, 256, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=128,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 512, 28, 28),
                weight_shape=(256, 512, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=256,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 1024, 14, 14),
                weight_shape=(512, 1024, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=512,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_dense_template(
                input_shape=(16, 1024),
                weight_shape=(1008, 1024),
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "densenet_121" and self.model_input == (1, 3, 240, 240) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern1(1, 7, 7, 960)
            pattern1(1, 7, 7, 928)
            pattern1(1, 7, 7, 896)
            pattern1(1, 7, 7, 864)
            pattern1(1, 7, 7, 832)
            pattern1(1, 7, 7, 800)
            pattern1(1, 7, 7, 768)
            pattern1(1, 7, 7, 736)
            pattern1(1, 7, 7, 704)
            pattern1(1, 7, 7, 672)
            pattern1(1, 7, 7, 640)
            pattern1(1, 7, 7, 608)
            pattern1(1, 7, 7, 576)
            pattern1(1, 7, 7, 544)
            pattern1(1, 7, 7, 512)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 128, 7, 7),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 16))
            
            pattern1(1, 15, 15, 992)
            pattern1(1, 15, 15, 960)
            pattern1(1, 15, 15, 928)
            pattern1(1, 15, 15, 896)
            pattern1(1, 15, 15, 864)
            pattern1(1, 15, 15, 832)
            pattern1(1, 15, 15, 800)
            pattern1(1, 15, 15, 768)
            pattern1(1, 15, 15, 736)
            pattern1(1, 15, 15, 704)
            pattern1(1, 15, 15, 672)
            pattern1(1, 15, 15, 640)
            pattern1(1, 15, 15, 608)
            pattern1(1, 15, 15, 576)
            pattern1(1, 15, 15, 544)
            pattern1(1, 15, 15, 512)
            pattern1(1, 15, 15, 480)
            pattern1(1, 15, 15, 448)
            pattern1(1, 15, 15, 416)
            pattern1(1, 15, 15, 384)
            pattern1(1, 15, 15, 352)
            pattern1(1, 15, 15, 320)
            pattern1(1, 15, 15, 288)
            pattern1(1, 15, 15, 256)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 128, 15, 15),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 24))
            
            pattern1(1, 30, 30, 480)
            pattern1(1, 30, 30, 448)
            pattern1(1, 30, 30, 384)
            pattern1(1, 30, 30, 352)
            pattern1(1, 30, 30, 320)      
            pattern1(1, 30, 30, 288)
            pattern1(1, 30, 30, 256)
            pattern1(1, 30, 30, 224)
            pattern1(1, 30, 30, 192)
            pattern1(1, 30, 30, 160)
            pattern1(1, 30, 30, 128)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 128, 30, 30),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            pattern1(1, 60, 60, 224)
            pattern1(1, 60, 60, 192)
            pattern1(1, 60, 60, 160)
            pattern1(1, 60, 60, 128)
            
            pattern1(1, 60, 60, 96)
            pattern1(1, 60, 60, 64)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 128, 60, 60),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 3, 240, 240),
                weight_shape=(64, 3, 7, 7),
                strides=(2, 2),
                padding=(3, 3),
                channels=64,
                kernel_size=(7, 7),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 256, 60, 60),
                weight_shape=(128, 256, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=128,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 512, 30, 30),
                weight_shape=(256, 512, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=256,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 1024, 15, 15),
                weight_shape=(512, 1024, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=512,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 1024),
                weight_shape=(1000, 1024),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "densenet_121" and self.model_input == (8, 3, 240, 240) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern1(8, 7, 7, 960)
            pattern1(8, 7, 7, 928)
            pattern1(8, 7, 7, 896)
            pattern1(8, 7, 7, 864)
            pattern1(8, 7, 7, 832)
            pattern1(8, 7, 7, 800)
            pattern1(8, 7, 7, 768)
            pattern1(8, 7, 7, 736)
            pattern1(8, 7, 7, 704)
            pattern1(8, 7, 7, 672)
            pattern1(8, 7, 7, 640)
            pattern1(8, 7, 7, 608)
            pattern1(8, 7, 7, 576)
            pattern1(8, 7, 7, 544)
            pattern1(8, 7, 7, 512)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 128, 7, 7),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 16))
            
            pattern1(8, 15, 15, 992)
            pattern1(8, 15, 15, 960)
            pattern1(8, 15, 15, 928)
            pattern1(8, 15, 15, 896)
            pattern1(8, 15, 15, 864)
            pattern1(8, 15, 15, 832)
            pattern1(8, 15, 15, 800)
            pattern1(8, 15, 15, 768)
            pattern1(8, 15, 15, 736)
            pattern1(8, 15, 15, 704)
            pattern1(8, 15, 15, 672)
            pattern1(8, 15, 15, 640)
            pattern1(8, 15, 15, 608)
            pattern1(8, 15, 15, 576)
            pattern1(8, 15, 15, 544)
            pattern1(8, 15, 15, 512)
            pattern1(8, 15, 15, 480)
            pattern1(8, 15, 15, 448)
            pattern1(8, 15, 15, 416)
            pattern1(8, 15, 15, 384)
            pattern1(8, 15, 15, 352)
            pattern1(8, 15, 15, 320)
            pattern1(8, 15, 15, 288)
            pattern1(8, 15, 15, 256)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 128, 15, 15),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 24))
            
            pattern1(8, 30, 30, 480)
            pattern1(8, 30, 30, 448)
            pattern1(8, 30, 30, 384)
            pattern1(8, 30, 30, 352)
            pattern1(8, 30, 30, 320)      
            pattern1(8, 30, 30, 288)
            pattern1(8, 30, 30, 256)
            pattern1(8, 30, 30, 224)
            pattern1(8, 30, 30, 192)
            pattern1(8, 30, 30, 160)
            pattern1(8, 30, 30, 128)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 128, 30, 30),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            pattern1(8, 60, 60, 224)
            pattern1(8, 60, 60, 192)
            pattern1(8, 60, 60, 160)
            pattern1(8, 60, 60, 128)
            pattern1(8, 60, 60, 96)
            pattern1(8, 60, 60, 64)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 128, 60, 60),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 3, 240, 240),
                weight_shape=(64, 3, 7, 7),
                strides=(2, 2),
                padding=(3, 3),
                channels=64,
                kernel_size=(7, 7),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 256, 60, 60),
                weight_shape=(128, 256, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=128,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 512, 30, 30),
                weight_shape=(256, 512, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=256,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 1024, 15, 15),
                weight_shape=(512, 1024, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=512,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_dense_template(
                input_shape=(8, 1024),
                weight_shape=(1024, 1024),
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "densenet_121" and self.model_input == (16, 3, 240, 240) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern1(16, 7, 7, 960)
            pattern1(16, 7, 7, 928)
            pattern1(16, 7, 7, 896)
            pattern1(16, 7, 7, 864)
            pattern1(16, 7, 7, 832)
            pattern1(16, 7, 7, 800)
            pattern1(16, 7, 7, 768)
            pattern1(16, 7, 7, 736)
            pattern1(16, 7, 7, 704)
            pattern1(16, 7, 7, 672)
            pattern1(16, 7, 7, 640)
            pattern1(16, 7, 7, 608)
            pattern1(16, 7, 7, 576)
            pattern1(16, 7, 7, 544)
            pattern1(16, 7, 7, 512)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 128, 7, 7),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 16))
            
            pattern1(16, 15, 15, 992)
            pattern1(16, 15, 15, 960)
            pattern1(16, 15, 15, 928)
            pattern1(16, 15, 15, 896)
            pattern1(16, 15, 15, 864)
            pattern1(16, 15, 15, 832)
            pattern1(16, 15, 15, 800)
            pattern1(16, 15, 15, 768)
            pattern1(16, 15, 15, 736)
            pattern1(16, 15, 15, 704)
            pattern1(16, 15, 15, 672)
            pattern1(16, 15, 15, 640)
            pattern1(16, 15, 15, 608)
            pattern1(16, 15, 15, 576)
            pattern1(16, 15, 15, 544)
            pattern1(16, 15, 15, 512)
            pattern1(16, 15, 15, 480)
            pattern1(16, 15, 15, 448)
            pattern1(16, 15, 15, 416)
            pattern1(16, 15, 15, 384)
            pattern1(16, 15, 15, 352)
            pattern1(16, 15, 15, 320)
            pattern1(16, 15, 15, 288)
            pattern1(16, 15, 15, 256)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 128, 15, 15),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 24))
            
            pattern1(16, 30, 30, 480)
            pattern1(16, 30, 30, 448)
            pattern1(16, 30, 30, 384)
            pattern1(16, 30, 30, 352)
            pattern1(16, 30, 30, 320)      
            pattern1(16, 30, 30, 288)
            pattern1(16, 30, 30, 256)
            pattern1(16, 30, 30, 224)
            pattern1(16, 30, 30, 192)
            pattern1(16, 30, 30, 160)
            pattern1(16, 30, 30, 128)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 128, 30, 30),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            pattern1(16, 60, 60, 224)
            pattern1(16, 60, 60, 192)
            pattern1(16, 60, 60, 160)
            pattern1(16, 60, 60, 128)
            pattern1(16, 60, 60, 96)
            pattern1(16, 60, 60, 64)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 128, 60, 60),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 3, 240, 240),
                weight_shape=(64, 3, 7, 7),
                strides=(2, 2),
                padding=(3, 3),
                channels=64,
                kernel_size=(7, 7),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 256, 60, 60),
                weight_shape=(128, 256, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=128,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 512, 30, 30),
                weight_shape=(256, 512, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=256,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 1024, 15, 15),
                weight_shape=(512, 1024, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=512,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_dense_template(
                input_shape=(16, 1024),
                weight_shape=(1008, 1024),
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "densenet_121" and self.model_input == (1, 3, 256, 256) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern1(1, 8, 8, 960)
            pattern1(1, 8, 8, 928)
            pattern1(1, 8, 8, 896)
            pattern1(1, 8, 8, 864)
            pattern1(1, 8, 8, 832)
            pattern1(1, 8, 8, 800)
            pattern1(1, 8, 8, 768)
            pattern1(1, 8, 8, 736)
            pattern1(1, 8, 8, 704)
            pattern1(1, 8, 8, 672)
            pattern1(1, 8, 8, 640)
            pattern1(1, 8, 8, 608)
            pattern1(1, 8, 8, 576)
            pattern1(1, 8, 8, 544)
            pattern1(1, 8, 8, 512)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 128, 8, 8),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 16))
            
            pattern1(1, 16, 16, 992)
            pattern1(1, 16, 16, 960)
            pattern1(1, 16, 16, 928)
            pattern1(1, 16, 16, 896)
            pattern1(1, 16, 16, 864)
            pattern1(1, 16, 16, 832)
            pattern1(1, 16, 16, 800)
            pattern1(1, 16, 16, 768)
            pattern1(1, 16, 16, 736)
            pattern1(1, 16, 16, 704)
            pattern1(1, 16, 16, 672)
            pattern1(1, 16, 16, 640)
            pattern1(1, 16, 16, 608)
            pattern1(1, 16, 16, 576)
            pattern1(1, 16, 16, 544)
            pattern1(1, 16, 16, 512)
            pattern1(1, 16, 16, 480)
            pattern1(1, 16, 16, 448)
            pattern1(1, 16, 16, 416)
            pattern1(1, 16, 16, 384)
            pattern1(1, 16, 16, 352)
            pattern1(1, 16, 16, 320)
            pattern1(1, 16, 16, 288)
            pattern1(1, 16, 16, 256)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 128, 16, 16),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 24))
            
            pattern1(1, 32, 32, 480)
            pattern1(1, 32, 32, 448)
            pattern1(1, 32, 32, 384)
            pattern1(1, 32, 32, 352)
            pattern1(1, 32, 32, 320)      
            pattern1(1, 32, 32, 288)
            pattern1(1, 32, 32, 256)
            pattern1(1, 32, 32, 224)
            pattern1(1, 32, 32, 192)
            pattern1(1, 32, 32, 160)
            pattern1(1, 32, 32, 128)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 128, 32, 32),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            pattern1(1, 64, 64, 224)
            pattern1(1, 64, 64, 192)
            pattern1(1, 64, 64, 160)
            pattern1(1, 64, 64, 128)
            pattern1(1, 64, 64, 96)
            pattern1(1, 64, 64, 64)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 128, 64, 64),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 3, 256, 256),
                weight_shape=(64, 3, 7, 7),
                strides=(2, 2),
                padding=(3, 3),
                channels=64,
                kernel_size=(7, 7),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 256, 64, 64),
                weight_shape=(128, 256, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=128,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 512, 32, 32),
                weight_shape=(256, 512, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=256,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
            param_dict = fused_nn_conv2d_template(
                input_shape=(1, 1024, 16, 16),
                weight_shape=(512, 1024, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=512,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 1024),
                weight_shape=(1000, 1024),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "densenet_121" and self.model_input == (8, 3, 256, 256) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern1(8, 8, 8, 960)
            pattern1(8, 8, 8, 928)
            pattern1(8, 8, 8, 896)
            pattern1(8, 8, 8, 864)
            pattern1(8, 8, 8, 832)
            pattern1(8, 8, 8, 800)
            pattern1(8, 8, 8, 768)
            pattern1(8, 8, 8, 736)
            pattern1(8, 8, 8, 704)
            pattern1(8, 8, 8, 672)
            pattern1(8, 8, 8, 640)
            pattern1(8, 8, 8, 608)
            pattern1(8, 8, 8, 576)
            pattern1(8, 8, 8, 544)
            pattern1(8, 8, 8, 512)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 128, 8, 8),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 16))
            
            
            pattern1(8, 16, 16, 992)
            pattern1(8, 16, 16, 960)
            pattern1(8, 16, 16, 928)
            pattern1(8, 16, 16, 896)
            pattern1(8, 16, 16, 864)
            pattern1(8, 16, 16, 832)
            pattern1(8, 16, 16, 800)
            pattern1(8, 16, 16, 768)
            pattern1(8, 16, 16, 736)
            pattern1(8, 16, 16, 704)
            pattern1(8, 16, 16, 672)
            pattern1(8, 16, 16, 640)
            pattern1(8, 16, 16, 608)
            pattern1(8, 16, 16, 576)
            pattern1(8, 16, 16, 544)
            pattern1(8, 16, 16, 512)
            pattern1(8, 16, 16, 480)
            pattern1(8, 16, 16, 448)
            pattern1(8, 16, 16, 416)
            pattern1(8, 16, 16, 384)
            pattern1(8, 16, 16, 352)
            pattern1(8, 16, 16, 320)
            pattern1(8, 16, 16, 288)
            pattern1(8, 16, 16, 256)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 128, 16, 16),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 24))
            
            pattern1(8, 32, 32, 480)
            pattern1(8, 32, 32, 448)
            pattern1(8, 32, 32, 384)
            pattern1(8, 32, 32, 352)
            pattern1(8, 32, 32, 320)      
            pattern1(8, 32, 32, 288)
            pattern1(8, 32, 32, 256)
            pattern1(8, 32, 32, 224)
            pattern1(8, 32, 32, 192)
            pattern1(8, 32, 32, 160)
            pattern1(8, 32, 32, 128)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 128, 32, 32),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            pattern1(8, 64, 64, 224)
            pattern1(8, 64, 64, 192)
            pattern1(8, 64, 64, 160)
            pattern1(8, 64, 64, 128)
            pattern1(8, 64, 64, 96)
            pattern1(8, 64, 64, 64)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 128, 64, 64),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 3, 256, 256),
                weight_shape=(64, 3, 7, 7),
                strides=(2, 2),
                padding=(3, 3),
                channels=64,
                kernel_size=(7, 7),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 256, 64, 64),
                weight_shape=(128, 256, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=128,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 512, 32, 32),
                weight_shape=(256, 512, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=256,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
            param_dict = fused_nn_conv2d_template(
                input_shape=(8, 1024, 16, 16),
                weight_shape=(512, 1024, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=512,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_dense_template(
                input_shape=(8, 1024),
                weight_shape=(1024, 1024)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "densenet_121" and self.model_input == (16, 3, 256, 256) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern1(16, 8, 8, 960)
            pattern1(16, 8, 8, 928)
            pattern1(16, 8, 8, 896)
            pattern1(16, 8, 8, 864)
            pattern1(16, 8, 8, 832)
            pattern1(16, 8, 8, 800)
            pattern1(16, 8, 8, 768)
            pattern1(16, 8, 8, 736)
            pattern1(16, 8, 8, 704)
            pattern1(16, 8, 8, 672)
            pattern1(16, 8, 8, 640)
            pattern1(16, 8, 8, 608)
            pattern1(16, 8, 8, 576)
            pattern1(16, 8, 8, 544)
            pattern1(16, 8, 8, 512)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 128, 8, 8),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 16))
            
            
            pattern1(16, 16, 16, 992)
            pattern1(16, 16, 16, 960)
            pattern1(16, 16, 16, 928)
            pattern1(16, 16, 16, 896)
            pattern1(16, 16, 16, 864)
            pattern1(16, 16, 16, 832)
            pattern1(16, 16, 16, 800)
            pattern1(16, 16, 16, 768)
            pattern1(16, 16, 16, 736)
            pattern1(16, 16, 16, 704)
            pattern1(16, 16, 16, 672)
            pattern1(16, 16, 16, 640)
            pattern1(16, 16, 16, 608)
            pattern1(16, 16, 16, 576)
            pattern1(16, 16, 16, 544)
            pattern1(16, 16, 16, 512)
            pattern1(16, 16, 16, 480)
            pattern1(16, 16, 16, 448)
            pattern1(16, 16, 16, 416)
            pattern1(16, 16, 16, 384)
            pattern1(16, 16, 16, 352)
            pattern1(16, 16, 16, 320)
            pattern1(16, 16, 16, 288)
            pattern1(16, 16, 16, 256)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 128, 16, 16),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 24))
            
            
            pattern1(16, 32, 32, 480)
            pattern1(16, 32, 32, 448)
            pattern1(16, 32, 32, 384)
            pattern1(16, 32, 32, 352)
            pattern1(16, 32, 32, 320)      
            pattern1(16, 32, 32, 288)
            pattern1(16, 32, 32, 256)
            pattern1(16, 32, 32, 224)
            pattern1(16, 32, 32, 192)
            pattern1(16, 32, 32, 160)
            pattern1(16, 32, 32, 128)
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 128, 32, 32),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            pattern1(16, 64, 64, 224)
            pattern1(16, 64, 64, 192)
            pattern1(16, 64, 64, 160)
            pattern1(16, 64, 64, 128)
            pattern1(16, 64, 64, 96)
            pattern1(16, 64, 64, 64)
            
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 128, 64, 64),
                weight_shape=(32, 128, 3, 3),
                strides=(1, 1),
                padding=(1, 1),
                channels=32,
                kernel_size=(3, 3),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 12))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 3, 256, 256),
                weight_shape=(64, 3, 7, 7),
                strides=(2, 2),
                padding=(3, 3),
                channels=64,
                kernel_size=(7, 7),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 256, 64, 64),
                weight_shape=(128, 256, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=128,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 512, 32, 32),
                weight_shape=(256, 512, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=256,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
            param_dict = fused_nn_conv2d_template(
                input_shape=(16, 1024, 16, 16),
                weight_shape=(512, 1024, 1, 1),
                strides=(1, 1),
                padding=(0, 0),
                channels=512,
                kernel_size=(1, 1),
                input_layout="NCHW",
                kernel_layout="OIHW",
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_conv2d', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            param_dict = fused_nn_dense_template(
                input_shape=(16, 1024),
                weight_shape=(1008, 1024)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        # BERT 
        elif self.name == "bert_base" and self.model_input == (1, 64) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 768),
                weight_shape=(768, 768),
                add_shape=(1, 768)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            pattern_bert(12, 64, 64, 64, 24)
            pattern_bert(1, 64, 768, 768, 48)
            pattern_bert(1, 64, 3072, 768, 12)
            pattern_bert(1, 64, 768, 3072, 12)
            

        elif self.name == "bert_base" and self.model_input == (1, 128) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 768),
                weight_shape=(768, 768),
                add_shape=(1, 768)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            pattern_bert(12, 128, 128, 64, 12)
            pattern_bert(12, 128, 64, 128, 12)
            pattern_bert(1, 128, 768, 768, 48)
            pattern_bert(1, 128, 3072, 768, 12)
            pattern_bert(1, 128, 768, 3072, 12)
            
            
        elif self.name == "bert_base" and self.model_input == (1, 256) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 768),
                weight_shape=(768, 768),
                add_shape=(1, 768)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(12, 256, 256, 64, 12)
            pattern_bert(12, 256, 64, 256, 12)
            pattern_bert(1, 256, 768, 768, 48)
            pattern_bert(1, 256, 3072, 768, 12)
            pattern_bert(1, 256, 768, 3072, 12)
            
            
        elif self.name == "bert_base" and self.model_input == (8, 64) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 768),
                weight_shape=(768, 768),
                add_shape=(1, 768)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            pattern_bert(96, 64, 64, 64, 24)
            pattern_bert(8, 64, 768, 768, 48)
            pattern_bert(8, 64, 3072, 768, 12)
            pattern_bert(8, 64, 768, 3072, 12)
            
            
        elif self.name == "bert_base" and self.model_input == (8, 128) and \
            self.model_dtype == "int8" and self.model_layout == "None":
                
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 768),
                weight_shape=(768, 768),
                add_shape=(1, 768)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            pattern_bert(96, 128, 128, 64, 12)
            pattern_bert(96, 128, 64, 128, 12)
            pattern_bert(8, 128, 768, 768, 48)
            pattern_bert(8, 128, 3072, 768, 12)
            pattern_bert(8, 128, 768, 3072, 12)
            
        elif self.name == "bert_base" and self.model_input == (8, 256) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 768),
                weight_shape=(768, 768),
                add_shape=(1, 768)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(96, 256, 256, 64, 12)
            pattern_bert(96, 256, 64, 256, 12)
            pattern_bert(8, 256, 768, 768, 48)
            pattern_bert(8, 256, 3072, 768, 12)
            pattern_bert(8, 256, 768, 3072, 12)
            
        elif self.name == "bert_base" and self.model_input == (16, 64) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 768),
                weight_shape=(768, 768),
                add_shape=(1, 768)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            pattern_bert(192, 64, 64, 64, 24)
            pattern_bert(16, 64, 768, 768, 48)
            pattern_bert(16, 64, 3072, 768, 12)
            pattern_bert(16, 64, 768, 3072, 12)
            
        elif self.name == "bert_base" and self.model_input == (16, 128) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 768),
                weight_shape=(768, 768),
                add_shape=(1, 768)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            pattern_bert(192, 128, 128, 64, 12)
            pattern_bert(192, 128, 64, 128, 12)
            pattern_bert(16, 128, 768, 768, 48)
            pattern_bert(16, 128, 3072, 768, 12)
            pattern_bert(16, 128, 768, 3072, 12)
        
        elif self.name == "bert_base" and self.model_input == (16, 256) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 768),
                weight_shape=(768, 768),
                add_shape=(1, 768)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(192, 256, 256, 64, 12)
            pattern_bert(192, 256, 64, 256, 12)
            pattern_bert(16, 256, 768, 768, 48)
            pattern_bert(16, 256, 3072, 768, 12)
            pattern_bert(16, 256, 768, 3072, 12)
        
        
        elif self.name == "bert_large" and self.model_input == (1, 64) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 1024),
                weight_shape=(1024, 1024),
                add_shape=(1, 1024)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(16, 64, 64, 64, 48)
            pattern_bert(1, 64, 1024, 1024, 96)
            pattern_bert(1, 64, 4096, 1024, 24)
            pattern_bert(1, 64, 1024, 4096, 24)
            
        elif self.name == "bert_large" and self.model_input == (1, 128) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 1024),
                weight_shape=(1024, 1024),
                add_shape=(1, 1024)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(16, 128, 128, 64, 24)
            pattern_bert(16, 128, 64, 128, 24)
            pattern_bert(1, 128, 1024, 1024, 96)
            pattern_bert(1, 128, 4096, 1024, 24)
            pattern_bert(1, 128, 1024, 4096, 24)
            
            
        elif self.name == "bert_large" and self.model_input == (1, 256) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 1024),
                weight_shape=(1024, 1024),
                add_shape=(1, 1024)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(16, 256, 256, 64, 24)
            pattern_bert(16, 256, 64, 256, 24)
            pattern_bert(1, 256, 1024, 1024, 96)
            pattern_bert(1, 256, 4096, 1024, 24)
            pattern_bert(1, 256, 1024, 4096, 24)
            
        
        elif self.name == "bert_large" and self.model_input == (8, 64) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 1024),
                weight_shape=(1024, 1024),
                add_shape=(1, 1024)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(128, 64, 64, 64, 48)
            pattern_bert(8, 64, 1024, 1024, 96)
            pattern_bert(8, 64, 4096, 1024, 24)
            pattern_bert(8, 64, 1024, 4096, 24)
        
        elif self.name == "bert_large" and self.model_input == (8, 128) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 1024),
                weight_shape=(1024, 1024),
                add_shape=(1, 1024)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(128, 128, 128, 64, 24)
            pattern_bert(128, 128, 64, 128, 24)
            pattern_bert(8, 128, 1024, 1024, 96)
            pattern_bert(8, 128, 4096, 1024, 24)
            pattern_bert(8, 128, 1024, 4096, 24)
            
        elif self.name == "bert_large" and self.model_input == (8, 256) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 1024),
                weight_shape=(1024, 1024),
                add_shape=(1, 1024)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(128, 256, 256, 64, 24)
            pattern_bert(128, 256, 64, 256, 24)
            pattern_bert(8, 256, 1024, 1024, 96)
            pattern_bert(8, 256, 4096, 1024, 24)
            pattern_bert(8, 256, 1024, 4096, 24)
        
        
        elif self.name == "bert_large" and self.model_input == (16, 64) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 1024),
                weight_shape=(1024, 1024),
                add_shape=(1, 1024)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(256, 64, 64, 64, 48)
            pattern_bert(16, 64, 1024, 1024, 96)
            pattern_bert(16, 64, 4096, 1024, 24)
            pattern_bert(16, 64, 1024, 4096, 24)
            
        elif self.name == "bert_large" and self.model_input == (16, 128) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 1024),
                weight_shape=(1024, 1024),
                add_shape=(1, 1024)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(256, 128, 128, 64, 24)
            pattern_bert(256, 128, 64, 128, 24)
            pattern_bert(16, 128, 1024, 1024, 96)
            pattern_bert(16, 128, 4096, 1024, 24)
            pattern_bert(16, 128, 1024, 4096, 24)
            
        elif self.name == "bert_large" and self.model_input == (16, 256) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 1024),
                weight_shape=(1024, 1024),
                add_shape=(1, 1024)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(256, 256, 256, 64, 24)
            pattern_bert(256, 256, 64, 256, 24)
            pattern_bert(16, 256, 1024, 1024, 96)
            pattern_bert(16, 256, 4096, 1024, 24)
            pattern_bert(16, 256, 1024, 4096, 24)
        
        elif self.name == "bert_medium" and self.model_input == (1, 64) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            pass
        elif self.name == "bert_medium" and self.model_input == (1, 128) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            pass
        elif self.name == "bert_medium" and self.model_input == (1, 256) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            pass
        elif self.name == "bert_medium" and self.model_input == (8, 64) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            pass
        elif self.name == "bert_medium" and self.model_input == (8, 128) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            pass
        elif self.name == "bert_medium" and self.model_input == (8, 256) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            pass
        elif self.name == "bert_medium" and self.model_input == (16, 64) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            pass
        elif self.name == "bert_medium" and self.model_input == (16, 128) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            pass
        elif self.name == "bert_medium" and self.model_input == (16, 256) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            pass
        
        elif self.name == "bert_tiny" and self.model_input == (1, 64) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 512),
                weight_shape=(512, 512),
                add_shape=(1, 512)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(8, 64, 64, 64, 12)
            pattern_bert(1, 64, 512, 512, 24)
            pattern_bert(1, 64, 2048, 512, 6)
            pattern_bert(1, 64, 512, 2048, 6)
            
            
        elif self.name == "bert_tiny" and self.model_input == (1, 128) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 512),
                weight_shape=(512, 512),
                add_shape=(1, 512)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(8, 128, 128, 64, 6)
            pattern_bert(8, 128, 64, 128, 6)
            pattern_bert(1, 128, 512, 512, 24)
            pattern_bert(1, 128, 2048, 512, 6)
            pattern_bert(1, 128, 512, 2048, 6)
            
        elif self.name == "bert_tiny" and self.model_input == (1, 256) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 512),
                weight_shape=(512, 512),
                add_shape=(1, 512)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(8, 256, 256, 64, 6)
            pattern_bert(8, 256, 64, 256, 6)
            pattern_bert(1, 256, 512, 512, 24)
            pattern_bert(1, 256, 2048, 512, 6)
            pattern_bert(1, 256, 512, 2048, 6)
        
        elif self.name == "bert_tiny" and self.model_input == (8, 64) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 512),
                weight_shape=(512, 512),
                add_shape=(1, 512)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(64, 64, 64, 64, 12)
            pattern_bert(8, 64, 512, 512, 24)
            pattern_bert(8, 64, 2048, 512, 6)
            pattern_bert(8, 64, 512, 2048, 6)
            
        elif self.name == "bert_tiny" and self.model_input == (8, 128) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 512),
                weight_shape=(512, 512),
                add_shape=(1, 512)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(64, 128, 128, 64, 6)
            pattern_bert(64, 128, 64, 128, 6)
            pattern_bert(8, 128, 512, 512, 24)
            pattern_bert(8, 128, 2048, 512, 6)
            pattern_bert(8, 128, 512, 2048, 6)
            
        elif self.name == "bert_tiny" and self.model_input == (8, 256) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 512),
                weight_shape=(512, 512),
                add_shape=(1, 512)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(64, 256, 256, 64, 6)
            pattern_bert(64, 256, 64, 256, 6)
            pattern_bert(8, 256, 512, 512, 24)
            pattern_bert(8, 256, 2048, 512, 6)
            pattern_bert(8, 256, 512, 2048, 6)
        
        
        elif self.name == "bert_tiny" and self.model_input == (16, 64) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 512),
                weight_shape=(512, 512),
                add_shape=(1, 512)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(128, 64, 64, 64, 12)
            pattern_bert(16, 64, 512, 512, 24)
            pattern_bert(16, 64, 2048, 512, 6)
            pattern_bert(16, 64, 512, 2048, 6)
            
        elif self.name == "bert_tiny" and self.model_input == (16, 128) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 512),
                weight_shape=(512, 512),
                add_shape=(1, 512)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(128, 128, 128, 64, 6)
            pattern_bert(128, 128, 64, 128, 6)
            pattern_bert(16, 128, 512, 512, 24)
            pattern_bert(16, 128, 2048, 512, 6)
            pattern_bert(16, 128, 512, 2048, 6)
            
        elif self.name == "bert_tiny" and self.model_input == (16, 256) and \
            self.model_dtype == "int8" and self.model_layout == "None":
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 512),
                weight_shape=(512, 512),
                add_shape=(1, 512)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            pattern_bert(128, 256, 256, 64, 6)
            pattern_bert(128, 256, 64, 256, 6)
            pattern_bert(16, 256, 512, 512, 24)
            pattern_bert(16, 256, 2048, 512, 6)
            pattern_bert(16, 256, 512, 2048, 6)
        
        # resnet18
        elif self.name == "resnet_18" and self.model_input == (1, 3, 224, 224) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(1, 1, 1, 256, 512, 14, 14, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 128, 256, 28, 28, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 64, 128, 56, 56, 2, 0, 1)
            pattern_resnet_add_relu(1, 7, 7, 3, 64, 224, 224, 2, 3, 1)
            pattern_resnet_add_relu(1, 3, 3, 64, 64, 56, 56, 1, 1, 2)
            pattern_resnet_add_add_relu(1, 3, 3, 64, 64, 56, 56, 1, 1, 2)
            pattern_resnet_add_relu(1, 3, 3, 64, 128, 56, 56, 2, 1, 1)
            pattern_resnet_add_relu(1, 3, 3, 128, 256, 28, 28, 2, 1, 1)
            pattern_resnet_add_relu(1, 3, 3, 256, 512, 14, 14, 2, 1, 1)
            pattern_resnet_add_add_relu(1, 3, 3, 512, 512, 7, 7, 1, 1, 1)
            pattern_resnet_add_relu(1, 3, 3, 512, 512, 7, 7, 1, 1, 1)
            # pattern_resnet_add_add_relu(1, 3, 3, 512, 512, 7, 7, 1, 1, 1)
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 512),
                weight_shape=(1000, 512),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
        elif self.name == "resnet_18" and self.model_input == (8, 3, 224, 224) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(8, 1, 1, 256, 512, 14, 14, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 128, 256, 28, 28, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 64, 128, 56, 56, 2, 0, 1)
            pattern_resnet_add_relu(8, 7, 7, 3, 64, 224, 224, 2, 3, 1)
            pattern_resnet_add_relu(8, 3, 3, 64, 64, 56, 56, 1, 1, 2)
            pattern_resnet_add_add_relu(8, 3, 3, 64, 64, 56, 56, 1, 1, 2)
            pattern_resnet_add_relu(8, 3, 3, 64, 128, 56, 56, 2, 1, 1)
            pattern_resnet_add_relu(8, 3, 3, 128, 256, 28, 28, 2, 1, 1)
            pattern_resnet_add_relu(8, 3, 3, 256, 512, 14, 14, 2, 1, 1)
            pattern_resnet_add_add_relu(8, 3, 3, 512, 512, 7, 7, 1, 1, 1)
            pattern_resnet_add_relu(8, 3, 3, 512, 512, 7, 7, 1, 1, 1)
            # pattern_resnet_add_add_relu(1, 3, 3, 512, 512, 7, 7, 1, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 512),
                weight_shape=(1000, 512),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
        elif self.name == "resnet_18" and self.model_input == (16, 3, 224, 224) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(16, 1, 1, 256, 512, 14, 14, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 128, 256, 28, 28, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 64, 128, 56, 56, 2, 0, 1)
            pattern_resnet_add_relu(16, 7, 7, 3, 64, 224, 224, 2, 3, 1)
            pattern_resnet_add_relu(16, 3, 3, 64, 64, 56, 56, 1, 1, 2)
            pattern_resnet_add_add_relu(16, 3, 3, 64, 64, 56, 56, 1, 1, 2)
            pattern_resnet_add_relu(16, 3, 3, 64, 128, 56, 56, 2, 1, 1)
            pattern_resnet_add_relu(16, 3, 3, 128, 256, 28, 28, 2, 1, 1)
            pattern_resnet_add_relu(16, 3, 3, 256, 512, 14, 14, 2, 1, 1)
            pattern_resnet_add_add_relu(16, 3, 3, 512, 512, 7, 7, 1, 1, 1)
            pattern_resnet_add_relu(16, 3, 3, 512, 512, 7, 7, 1, 1, 1)
            # pattern_resnet_add_add_relu(1, 3, 3, 512, 512, 7, 7, 1, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 512),
                weight_shape=(1000, 512),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "resnet_18" and self.model_input == (1, 3, 240, 240) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(1, 1, 1, 256, 512, 15, 15, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 128, 256, 30, 30, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 64, 128, 60, 60, 2, 0, 1)
            pattern_resnet_add_relu(1, 7, 7, 3, 64, 240, 240, 2, 3, 1)
            pattern_resnet_add_relu(1, 3, 3, 64, 64, 60, 60, 1, 1, 2)
            pattern_resnet_add_add_relu(1, 3, 3, 64, 64, 60, 60, 1, 1, 2)
            pattern_resnet_add_relu(1, 3, 3, 64, 128, 60, 60, 2, 1, 1)
            pattern_resnet_add_relu(1, 3, 3, 128, 256, 30, 30, 2, 1, 1)
            pattern_resnet_add_relu(1, 3, 3, 256, 512, 15, 15, 2, 1, 1)
            pattern_resnet_add_add_relu(1, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            pattern_resnet_add_relu(1, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            # pattern_resnet_add_add_relu(1, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 512),
                weight_shape=(1000, 512),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "resnet_18" and self.model_input == (8, 3, 240, 240) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(8, 1, 1, 256, 512, 15, 15, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 128, 256, 30, 30, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 64, 128, 60, 60, 2, 0, 1)
            pattern_resnet_add_relu(8, 7, 7, 3, 64, 240, 240, 2, 3, 1)
            pattern_resnet_add_relu(8, 3, 3, 64, 64, 60, 60, 1, 1, 2)
            pattern_resnet_add_add_relu(8, 3, 3, 64, 64, 60, 60, 1, 1, 2)
            pattern_resnet_add_relu(8, 3, 3, 64, 128, 60, 60, 2, 1, 1)
            pattern_resnet_add_relu(8, 3, 3, 128, 256, 30, 30, 2, 1, 1)
            pattern_resnet_add_relu(8, 3, 3, 256, 512, 15, 15, 2, 1, 1)
            pattern_resnet_add_add_relu(8, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            pattern_resnet_add_relu(8, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            # pattern_resnet_add_add_relu(1, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 512),
                weight_shape=(1000, 512),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
        elif self.name == "resnet_18" and self.model_input == (16, 3, 240, 240) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(16, 1, 1, 256, 512, 15, 15, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 128, 256, 30, 30, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 64, 128, 60, 60, 2, 0, 1)
            pattern_resnet_add_relu(16, 7, 7, 3, 64, 240, 240, 2, 3, 1)
            pattern_resnet_add_relu(16, 3, 3, 64, 64, 60, 60, 1, 1, 2)
            pattern_resnet_add_add_relu(16, 3, 3, 64, 64, 60, 60, 1, 1, 2)
            pattern_resnet_add_relu(16, 3, 3, 64, 128, 60, 60, 2, 1, 1)
            pattern_resnet_add_relu(16, 3, 3, 128, 256, 30, 30, 2, 1, 1)
            pattern_resnet_add_relu(16, 3, 3, 256, 512, 15, 15, 2, 1, 1)
            pattern_resnet_add_add_relu(16, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            pattern_resnet_add_relu(16, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            # pattern_resnet_add_add_relu(1, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 512),
                weight_shape=(1000, 512),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "resnet_18" and self.model_input == (1, 3, 256, 256) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(1, 1, 1, 256, 512, 16, 16, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 128, 256, 32, 32, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 64, 128, 64, 64, 2, 0, 1)
            pattern_resnet_add_relu(1, 7, 7, 3, 64, 256, 256, 2, 3, 1)
            pattern_resnet_add_relu(1, 3, 3, 64, 64, 64, 64, 1, 1, 2)
            pattern_resnet_add_add_relu(1, 3, 3, 64, 64, 64, 64, 1, 1, 2)
            pattern_resnet_add_relu(1, 3, 3, 64, 128, 64, 64, 2, 1, 1)
            pattern_resnet_add_relu(1, 3, 3, 128, 256, 32, 32, 2, 1, 1)
            pattern_resnet_add_relu(1, 3, 3, 256, 512, 16, 16, 2, 1, 1)
            pattern_resnet_add_add_relu(1, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            pattern_resnet_add_relu(1, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            # pattern_resnet_add_add_relu(1, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 512),
                weight_shape=(1000, 512),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            
        elif self.name == "resnet_18" and self.model_input == (8, 3, 256, 256) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(8, 1, 1, 256, 512, 16, 16, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 128, 256, 32, 32, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 64, 128, 64, 64, 2, 0, 1)
            pattern_resnet_add_relu(8, 7, 7, 3, 64, 256, 256, 2, 3, 1)
            pattern_resnet_add_relu(8, 3, 3, 64, 64, 64, 64, 1, 1, 2)
            pattern_resnet_add_add_relu(8, 3, 3, 64, 64, 64, 64, 1, 1, 2)
            pattern_resnet_add_relu(8, 3, 3, 64, 128, 64, 64, 2, 1, 1)
            pattern_resnet_add_relu(8, 3, 3, 128, 256, 32, 32, 2, 1, 1)
            pattern_resnet_add_relu(8, 3, 3, 256, 512, 16, 16, 2, 1, 1)
            pattern_resnet_add_add_relu(8, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            pattern_resnet_add_relu(8, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            # pattern_resnet_add_add_relu(1, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 512),
                weight_shape=(1000, 512),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "resnet_18" and self.model_input == (16, 3, 256, 256) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(16, 1, 1, 256, 512, 16, 16, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 128, 256, 32, 32, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 64, 128, 64, 64, 2, 0, 1)
            pattern_resnet_add_relu(16, 7, 7, 3, 64, 256, 256, 2, 3, 1)
            pattern_resnet_add_relu(16, 3, 3, 64, 64, 64, 64, 1, 1, 2)
            pattern_resnet_add_add_relu(16, 3, 3, 64, 64, 64, 64, 1, 1, 2)
            pattern_resnet_add_relu(16, 3, 3, 64, 128, 64, 64, 2, 1, 1)
            pattern_resnet_add_relu(16, 3, 3, 128, 256, 32, 32, 2, 1, 1)
            pattern_resnet_add_relu(16, 3, 3, 256, 512, 16, 16, 2, 1, 1)
            pattern_resnet_add_add_relu(16, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            pattern_resnet_add_relu(16, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            # pattern_resnet_add_add_relu(1, 3, 3, 512, 512, 8, 8, 1, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 512),
                weight_shape=(1000, 512),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        
        # resnet 50
        elif self.name == "resnet_50" and self.model_input == (1, 3, 224, 224) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(1, 1, 1, 1024, 2048, 14, 14, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 512, 1024, 28, 28, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 256, 512, 56, 56, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 64, 256, 56, 56, 1, 0, 1)
            pattern_resnet_add_relu(1, 7, 7, 3, 64, 224, 224, 2, 3, 1)
            pattern_resnet_add_relu(1, 1, 1, 64, 64, 56, 56, 1, 0, 1)
            pattern_resnet_add_relu(1, 1, 1, 256, 64, 56, 56, 1, 0, 2)
            pattern_resnet_add_relu(1, 3, 3, 64, 64, 56, 56, 1, 1, 3)
            pattern_resnet_add_add_relu(1, 1, 1, 64, 256, 56, 56, 1, 0, 3)
            pattern_resnet_add_relu(1, 1, 1, 256, 128, 56, 56, 1, 0, 1)
            pattern_resnet_add_relu(1, 3, 3, 128, 128, 56, 56, 2, 1, 1)
            pattern_resnet_add_relu(1, 1, 1, 512, 128, 28, 28, 1, 0, 3)
            pattern_resnet_add_add_relu(1, 1, 1, 128, 512, 28, 28, 1, 0, 4)
            pattern_resnet_add_relu(1, 1, 1, 512, 256, 28, 28, 1, 0, 1)
            pattern_resnet_add_relu(1, 3, 3, 256, 256, 28, 28, 2, 1, 1)
            pattern_resnet_add_relu(1, 1, 1, 1024, 256, 14, 14, 1, 0, 5)
            pattern_resnet_add_add_relu(1, 1, 1, 256, 1024, 14, 14, 1, 0, 6)
            pattern_resnet_add_relu(1, 1, 1, 1024, 512, 14, 14, 1, 0, 1)
            pattern_resnet_add_relu(1, 3, 3, 512, 512, 14, 14, 2, 1, 1)
            pattern_resnet_add_add_relu(1, 1, 1, 512, 2048, 7, 7, 1, 0, 2)
            pattern_resnet_add_relu(1, 1, 1, 2048, 512, 7, 7, 1, 0, 2)
            pattern_resnet_add_relu(1, 3, 3, 512, 512, 7, 7, 1, 1, 2)
            # pattern_resnet_add_add_relu(1, 1, 1, 512, 2048, 7, 7, 1, 0, 1)
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 2048),
                weight_shape=(1000, 2048),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            
        elif self.name == "resnet_50" and self.model_input == (8, 3, 224, 224) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(8, 1, 1, 1024, 2048, 14, 14, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 512, 1024, 28, 28, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 256, 512, 56, 56, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 64, 256, 56, 56, 1, 0, 1)
            pattern_resnet_add_relu(8, 7, 7, 3, 64, 224, 224, 2, 3, 1)
            pattern_resnet_add_relu(8, 1, 1, 64, 64, 56, 56, 1, 0, 1)
            pattern_resnet_add_relu(8, 1, 1, 256, 64, 56, 56, 1, 0, 2)
            pattern_resnet_add_relu(8, 3, 3, 64, 64, 56, 56, 1, 1, 3)
            pattern_resnet_add_add_relu(8, 1, 1, 64, 256, 56, 56, 1, 0, 3)
            pattern_resnet_add_relu(8, 1, 1, 256, 128, 56, 56, 1, 0, 1)
            pattern_resnet_add_relu(8, 3, 3, 128, 128, 56, 56, 2, 1, 1)
            pattern_resnet_add_relu(8, 1, 1, 512, 128, 28, 28, 1, 0, 3)
            pattern_resnet_add_add_relu(8, 1, 1, 128, 512, 28, 28, 1, 0, 4)
            pattern_resnet_add_relu(8, 1, 1, 512, 256, 28, 28, 1, 0, 1)
            pattern_resnet_add_relu(8, 3, 3, 256, 256, 28, 28, 2, 1, 1)
            pattern_resnet_add_relu(8, 1, 1, 1024, 256, 14, 14, 1, 0, 5)
            pattern_resnet_add_add_relu(8, 1, 1, 256, 1024, 14, 14, 1, 0, 6)
            pattern_resnet_add_relu(8, 1, 1, 1024, 512, 14, 14, 1, 0, 1)
            pattern_resnet_add_relu(8, 3, 3, 512, 512, 14, 14, 2, 1, 1)
            pattern_resnet_add_add_relu(8, 1, 1, 512, 2048, 7, 7, 1, 0, 2)
            pattern_resnet_add_relu(8, 1, 1, 2048, 512, 7, 7, 1, 0, 2)
            pattern_resnet_add_relu(8, 3, 3, 512, 512, 7, 7, 1, 1, 2)
            # pattern_resnet_add_add_relu(1, 1, 1, 512, 2048, 7, 7, 1, 0, 1)
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 2048),
                weight_shape=(1000, 2048),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "resnet_50" and self.model_input == (16, 3, 224, 224) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(16, 1, 1, 1024, 2048, 14, 14, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 512, 1024, 28, 28, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 256, 512, 56, 56, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 64, 256, 56, 56, 1, 0, 1)
            pattern_resnet_add_relu(16, 7, 7, 3, 64, 224, 224, 2, 3, 1)
            pattern_resnet_add_relu(16, 1, 1, 64, 64, 56, 56, 1, 0, 1)
            pattern_resnet_add_relu(16, 1, 1, 256, 64, 56, 56, 1, 0, 2)
            pattern_resnet_add_relu(16, 3, 3, 64, 64, 56, 56, 1, 1, 3)
            pattern_resnet_add_add_relu(16, 1, 1, 64, 256, 56, 56, 1, 0, 3)
            pattern_resnet_add_relu(16, 1, 1, 256, 128, 56, 56, 1, 0, 1)
            pattern_resnet_add_relu(16, 3, 3, 128, 128, 56, 56, 2, 1, 1)
            pattern_resnet_add_relu(16, 1, 1, 512, 128, 28, 28, 1, 0, 3)
            pattern_resnet_add_add_relu(6, 1, 1, 128, 512, 28, 28, 1, 0, 4)
            pattern_resnet_add_relu(16, 1, 1, 512, 256, 28, 28, 1, 0, 1)
            pattern_resnet_add_relu(16, 3, 3, 256, 256, 28, 28, 2, 1, 1)
            pattern_resnet_add_relu(16, 1, 1, 1024, 256, 14, 14, 1, 0, 5)
            pattern_resnet_add_add_relu(16, 1, 1, 256, 1024, 14, 14, 1, 0, 6)
            pattern_resnet_add_relu(16, 1, 1, 1024, 512, 14, 14, 1, 0, 1)
            pattern_resnet_add_relu(16, 3, 3, 512, 512, 14, 14, 2, 1, 1)
            pattern_resnet_add_add_relu(16, 1, 1, 512, 2048, 7, 7, 1, 0, 2)
            pattern_resnet_add_relu(16, 1, 1, 2048, 512, 7, 7, 1, 0, 2)
            pattern_resnet_add_relu(16, 3, 3, 512, 512, 7, 7, 1, 1, 2)
            # pattern_resnet_add_add_relu(1, 1, 1, 512, 2048, 7, 7, 1, 0, 1)
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 2048),
                weight_shape=(1000, 2048),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "resnet_50" and self.model_input == (1, 3, 240, 240) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(1, 1, 1, 1024, 2048, 15, 15, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 512, 1024, 30, 30, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 256, 512, 60, 60, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 64, 256, 60, 60, 1, 0, 1)
            pattern_resnet_add_relu(1, 7, 7, 3, 64, 240, 240, 2, 3, 1)
            pattern_resnet_add_relu(1, 1, 1, 64, 64, 60, 60, 1, 0, 1)
            pattern_resnet_add_relu(1, 1, 1, 256, 64, 60, 60, 1, 0, 2)
            pattern_resnet_add_relu(1, 3, 3, 64, 64, 60, 60, 1, 1, 3)
            pattern_resnet_add_add_relu(1, 1, 1, 64, 256, 60, 60, 1, 0, 3)
            pattern_resnet_add_relu(1, 1, 1, 256, 128, 60, 60, 1, 0, 1)
            pattern_resnet_add_relu(1, 3, 3, 128, 128, 60, 60, 2, 1, 1)
            pattern_resnet_add_relu(1, 1, 1, 512, 128, 30, 30, 1, 0, 3)
            pattern_resnet_add_add_relu(1, 1, 1, 128, 512, 30, 30, 1, 0, 4)
            pattern_resnet_add_relu(1, 1, 1, 512, 256, 30, 30, 1, 0, 1)
            pattern_resnet_add_relu(1, 3, 3, 256, 256, 30, 30, 2, 1, 1)
            pattern_resnet_add_relu(1, 1, 1, 1024, 256, 15, 15, 1, 0, 5)
            pattern_resnet_add_add_relu(1, 1, 1, 256, 1024, 15, 15, 1, 0, 6)
            pattern_resnet_add_relu(1, 1, 1, 1024, 512, 15, 15, 1, 0, 1)
            pattern_resnet_add_relu(1, 3, 3, 512, 512, 15, 15, 2, 1, 1)
            pattern_resnet_add_add_relu(1, 1, 1, 512, 2048, 8, 8, 1, 0, 2)
            pattern_resnet_add_relu(1, 1, 1, 2048, 512, 8, 8, 1, 0, 2)
            pattern_resnet_add_relu(1, 3, 3, 512, 512, 8, 8, 1, 1, 2)
            # pattern_resnet_add_add_relu(1, 1, 1, 512, 2048, 7, 7, 1, 0, 1)
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 2048),
                weight_shape=(1000, 2048),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "resnet_50" and self.model_input == (8, 3, 240, 240) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(8, 1, 1, 1024, 2048, 15, 15, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 512, 1024, 30, 30, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 256, 512, 60, 60, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 64, 256, 60, 60, 1, 0, 1)
            pattern_resnet_add_relu(8, 7, 7, 3, 64, 240, 240, 2, 3, 1)
            pattern_resnet_add_relu(8, 1, 1, 64, 64, 60, 60, 1, 0, 1)
            pattern_resnet_add_relu(8, 1, 1, 256, 64, 60, 60, 1, 0, 2)
            pattern_resnet_add_relu(8, 3, 3, 64, 64, 60, 60, 1, 1, 3)
            pattern_resnet_add_add_relu(8, 1, 1, 64, 256, 60, 60, 1, 0, 3)
            pattern_resnet_add_relu(8, 1, 1, 256, 128, 60, 60, 1, 0, 1)
            pattern_resnet_add_relu(8, 3, 3, 128, 128, 60, 60, 2, 1, 1)
            pattern_resnet_add_relu(8, 1, 1, 512, 128, 30, 30, 1, 0, 3)
            pattern_resnet_add_add_relu(8, 1, 1, 128, 512, 30, 30, 1, 0, 4)
            pattern_resnet_add_relu(8, 1, 1, 512, 256, 30, 30, 1, 0, 1)
            pattern_resnet_add_relu(8, 3, 3, 256, 256, 30, 30, 2, 1, 1)
            pattern_resnet_add_relu(8, 1, 1, 1024, 256, 15, 15, 1, 0, 5)
            pattern_resnet_add_add_relu(8, 1, 1, 256, 1024, 15, 15, 1, 0, 6)
            pattern_resnet_add_relu(8, 1, 1, 1024, 512, 15, 15, 1, 0, 1)
            pattern_resnet_add_relu(8, 3, 3, 512, 512, 15, 15, 2, 1, 1)
            pattern_resnet_add_add_relu(8, 1, 1, 512, 2048, 8, 8, 1, 0, 2)
            pattern_resnet_add_relu(8, 1, 1, 2048, 512, 8, 8, 1, 0, 2)
            pattern_resnet_add_relu(8, 3, 3, 512, 512, 8, 8, 1, 1, 2)
            # pattern_resnet_add_add_relu(1, 1, 1, 512, 2048, 7, 7, 1, 0, 1)
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 2048),
                weight_shape=(1000, 2048),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "resnet_50" and self.model_input == (16, 3, 240, 240) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(16, 1, 1, 1024, 2048, 15, 15, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 512, 1024, 30, 30, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 256, 512, 60, 60, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 64, 256, 60, 60, 1, 0, 1)
            pattern_resnet_add_relu(16, 7, 7, 3, 64, 240, 240, 2, 3, 1)
            pattern_resnet_add_relu(16, 1, 1, 64, 64, 60, 60, 1, 0, 1)
            pattern_resnet_add_relu(16, 1, 1, 256, 64, 60, 60, 1, 0, 2)
            pattern_resnet_add_relu(16, 3, 3, 64, 64, 60, 60, 1, 1, 3)
            pattern_resnet_add_add_relu(16, 1, 1, 64, 256, 60, 60, 1, 0, 3)
            pattern_resnet_add_relu(16, 1, 1, 256, 128, 60, 60, 1, 0, 1)
            pattern_resnet_add_relu(16, 3, 3, 128, 128, 60, 60, 2, 1, 1)
            pattern_resnet_add_relu(16, 1, 1, 512, 128, 30, 30, 1, 0, 3)
            pattern_resnet_add_add_relu(16, 1, 1, 128, 512, 30, 30, 1, 0, 4)
            pattern_resnet_add_relu(16, 1, 1, 512, 256, 30, 30, 1, 0, 1)
            pattern_resnet_add_relu(16, 3, 3, 256, 256, 30, 30, 2, 1, 1)
            pattern_resnet_add_relu(16, 1, 1, 1024, 256, 15, 15, 1, 0, 5)
            pattern_resnet_add_add_relu(16, 1, 1, 256, 1024, 15, 15, 1, 0, 6)
            pattern_resnet_add_relu(16, 1, 1, 1024, 512, 15, 15, 1, 0, 1)
            pattern_resnet_add_relu(16, 3, 3, 512, 512, 15, 15, 2, 1, 1)
            pattern_resnet_add_add_relu(16, 1, 1, 512, 2048, 8, 8, 1, 0, 2)
            pattern_resnet_add_relu(16, 1, 1, 2048, 512, 8, 8, 1, 0, 2)
            pattern_resnet_add_relu(16, 3, 3, 512, 512, 8, 8, 1, 1, 2)
            # pattern_resnet_add_add_relu(1, 1, 1, 512, 2048, 7, 7, 1, 0, 1)
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 2048),
                weight_shape=(1000, 2048),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "resnet_50" and self.model_input == (1, 3, 256, 256) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(1, 1, 1, 1024, 2048, 16, 16, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 512, 1024, 32, 32, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 256, 512, 64, 64, 2, 0, 1)
            pattern_resnet_add(1, 1, 1, 64, 256, 64, 64, 1, 0, 1)
            pattern_resnet_add_relu(1, 7, 7, 3, 64, 256, 256, 2, 3, 1)
            pattern_resnet_add_relu(1, 1, 1, 64, 64, 64, 64, 1, 0, 1)
            pattern_resnet_add_relu(1, 1, 1, 256, 64, 64, 64, 1, 0, 2)
            pattern_resnet_add_relu(1, 3, 3, 64, 64, 64, 64, 1, 1, 3)
            pattern_resnet_add_add_relu(1, 1, 1, 64, 256, 64, 64, 1, 0, 3)
            pattern_resnet_add_relu(1, 1, 1, 256, 128, 64, 64, 1, 0, 1)
            pattern_resnet_add_relu(1, 3, 3, 128, 128, 64, 64, 2, 1, 1)
            pattern_resnet_add_relu(1, 1, 1, 512, 128, 32, 32, 1, 0, 3)
            pattern_resnet_add_add_relu(1, 1, 1, 128, 512, 32, 32, 1, 0, 4)
            pattern_resnet_add_relu(1, 1, 1, 512, 256, 32, 32, 1, 0, 1)
            pattern_resnet_add_relu(1, 3, 3, 256, 256, 32, 32, 2, 1, 1)
            pattern_resnet_add_relu(1, 1, 1, 1024, 256, 16, 16, 1, 0, 5)
            pattern_resnet_add_add_relu(1, 1, 1, 256, 1024, 16, 16, 1, 0, 6)
            pattern_resnet_add_relu(1, 1, 1, 1024, 512, 16, 16, 1, 0, 1)
            pattern_resnet_add_relu(1, 3, 3, 512, 512, 16, 16, 2, 1, 1)
            pattern_resnet_add_add_relu(1, 1, 1, 512, 2048, 8, 8, 1, 0, 2)
            pattern_resnet_add_relu(1, 1, 1, 2048, 512, 8, 8, 1, 0, 2)
            pattern_resnet_add_relu(1, 3, 3, 512, 512, 8, 8, 1, 1, 2)
            # pattern_resnet_add_add_relu(1, 1, 1, 512, 2048, 7, 7, 1, 0, 1)
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 2048),
                weight_shape=(1000, 2048),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "resnet_50" and self.model_input == (8, 3, 256, 256) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(8, 1, 1, 1024, 2048, 16, 16, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 512, 1024, 32, 32, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 256, 512, 64, 64, 2, 0, 1)
            pattern_resnet_add(8, 1, 1, 64, 256, 64, 64, 1, 0, 1)
            pattern_resnet_add_relu(8, 7, 7, 3, 64, 256, 256, 2, 3, 1)
            pattern_resnet_add_relu(8, 1, 1, 64, 64, 64, 64, 1, 0, 1)
            pattern_resnet_add_relu(8, 1, 1, 256, 64, 64, 64, 1, 0, 2)
            pattern_resnet_add_relu(8, 3, 3, 64, 64, 64, 64, 1, 1, 3)
            pattern_resnet_add_add_relu(8, 1, 1, 64, 256, 64, 64, 1, 0, 3)
            pattern_resnet_add_relu(8, 1, 1, 256, 128, 64, 64, 1, 0, 1)
            pattern_resnet_add_relu(8, 3, 3, 128, 128, 64, 64, 2, 1, 1)
            pattern_resnet_add_relu(8, 1, 1, 512, 128, 32, 32, 1, 0, 3)
            pattern_resnet_add_add_relu(8, 1, 1, 128, 512, 32, 32, 1, 0, 4)
            pattern_resnet_add_relu(8, 1, 1, 512, 256, 32, 32, 1, 0, 1)
            pattern_resnet_add_relu(8, 3, 3, 256, 256, 32, 32, 2, 1, 1)
            pattern_resnet_add_relu(8, 1, 1, 1024, 256, 16, 16, 1, 0, 5)
            pattern_resnet_add_add_relu(8, 1, 1, 256, 1024, 16, 16, 1, 0, 6)
            pattern_resnet_add_relu(8, 1, 1, 1024, 512, 16, 16, 1, 0, 1)
            pattern_resnet_add_relu(8, 3, 3, 512, 512, 16, 16, 2, 1, 1)
            pattern_resnet_add_add_relu(8, 1, 1, 512, 2048, 8, 8, 1, 0, 2)
            pattern_resnet_add_relu(8, 1, 1, 2048, 512, 8, 8, 1, 0, 2)
            pattern_resnet_add_relu(8, 3, 3, 512, 512, 8, 8, 1, 1, 2)
            # pattern_resnet_add_add_relu(1, 1, 1, 512, 2048, 7, 7, 1, 0, 1)
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 2048),
                weight_shape=(1000, 2048),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "resnet_50" and self.model_input == (16, 3, 256, 256) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_resnet_add(16, 1, 1, 1024, 2048, 16, 16, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 512, 1024, 32, 32, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 256, 512, 64, 64, 2, 0, 1)
            pattern_resnet_add(16, 1, 1, 64, 256, 64, 64, 1, 0, 1)
            pattern_resnet_add_relu(16, 7, 7, 3, 64, 256, 256, 2, 3, 1)
            pattern_resnet_add_relu(16, 1, 1, 64, 64, 64, 64, 1, 0, 1)
            pattern_resnet_add_relu(16, 1, 1, 256, 64, 64, 64, 1, 0, 2)
            pattern_resnet_add_relu(16, 3, 3, 64, 64, 64, 64, 1, 1, 3)
            pattern_resnet_add_add_relu(16, 1, 1, 64, 256, 64, 64, 1, 0, 3)
            pattern_resnet_add_relu(16, 1, 1, 256, 128, 64, 64, 1, 0, 1)
            pattern_resnet_add_relu(16, 3, 3, 128, 128, 64, 64, 2, 1, 1)
            pattern_resnet_add_relu(16, 1, 1, 512, 128, 32, 32, 1, 0, 3)
            pattern_resnet_add_add_relu(16, 1, 1, 128, 512, 32, 32, 1, 0, 4)
            pattern_resnet_add_relu(16, 1, 1, 512, 256, 32, 32, 1, 0, 1)
            pattern_resnet_add_relu(16, 3, 3, 256, 256, 32, 32, 2, 1, 1)
            pattern_resnet_add_relu(16, 1, 1, 1024, 256, 16, 16, 1, 0, 5)
            pattern_resnet_add_add_relu(16, 1, 1, 256, 1024, 16, 16, 1, 0, 6)
            pattern_resnet_add_relu(16, 1, 1, 1024, 512, 16, 16, 1, 0, 1)
            pattern_resnet_add_relu(16, 3, 3, 512, 512, 16, 16, 2, 1, 1)
            pattern_resnet_add_add_relu(16, 1, 1, 512, 2048, 8, 8, 1, 0, 2)
            pattern_resnet_add_relu(16, 1, 1, 2048, 512, 8, 8, 1, 0, 2)
            pattern_resnet_add_relu(16, 3, 3, 512, 512, 8, 8, 1, 1, 2)
            # pattern_resnet_add_add_relu(1, 1, 1, 512, 2048, 7, 7, 1, 0, 1)
            
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 2048),
                weight_shape=(1000, 2048),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "inception_v3" and self.model_input == (1, 3, 299, 299) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_inception_v3_add_relu(1, 1, 1, 2048, 192, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 1024, 320, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 2048, 448, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 128, 192, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 1280, 320, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 1280, 448, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 3, 1, 384, 384, 8, 8, 1, (1, 0, 1, 0), 4)
            pattern_inception_v3_add_relu(1, 3, 3, 192, 192, 17, 17, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 7, 192, 192, 17, 17, 1, (0, 3, 0, 3), 4)
            pattern_inception_v3_add_relu(1, 7, 1, 192, 192, 17, 17, 1, (3, 0, 3, 0), 4)
            pattern_inception_v3_add_relu(1, 7, 1, 160, 160, 17, 17, 1, (3, 0, 3, 0), 4)
            pattern_inception_v3_add_relu(1, 1, 7, 160, 192, 17, 17, 1, (0, 3, 0, 3), 2)
            pattern_inception_v3_add_relu(1, 1, 1, 768, 160, 17, 17, 1, (0, 0, 0, 0), 4)
            pattern_inception_v3_add_relu(1, 1, 7, 160, 160, 17, 17, 1, (0, 3, 0, 3), 4)
            pattern_inception_v3_add_relu(1, 7, 1, 160, 192, 17, 17, 1, (3, 0, 3, 0), 2)
            pattern_inception_v3_add_relu(1, 7, 1, 128, 128, 17, 17, 1, (3, 0, 3, 0), 2)
            pattern_inception_v3_add_relu(1, 1, 7, 128, 192, 17, 17, 1, (0, 3, 0, 3), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 768, 128, 17, 17, 1, (0, 0, 0, 0), 2)
            pattern_inception_v3_add_relu(1, 1, 7, 128, 128, 17, 17, 1, (0, 3, 0, 3), 2)
            pattern_inception_v3_add_relu(1, 7, 1, 128, 192, 17, 17, 1, (3, 0, 3, 0), 1)
            pattern_inception_v3_add_relu(1, 3, 3, 96, 96, 35, 35, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 288, 48, 35, 35, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 256, 48, 35, 35, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 192, 32, 35, 35, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 192, 48, 35, 35, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 5, 5, 48, 46, 35, 35, 1, (2, 2, 2, 2), 3)
            pattern_inception_v3_add_relu(1, 3, 3, 3, 32, 299, 299, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 3, 3, 32, 32, 149, 149, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 3, 3, 32, 64, 147, 147, 1, (1, 1, 1, 1), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 64, 80, 73, 73, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 192, 64, 35, 35, 1, (0, 0, 0, 0), 2)
            pattern_inception_v3_add_relu(1, 1, 1, 256, 64, 35, 35, 1, (0, 0, 0, 0), 3)
            pattern_inception_v3_add_relu(1, 1, 1, 288, 64, 35, 35, 1, (0, 0, 0, 0,), 4)
            pattern_inception_v3_add_relu(1, 3, 3, 288, 384, 35, 35, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 768, 192, 17, 17, 1, (0, 0, 0, 0), 12)
            pattern_inception_v3_add_relu(1, 3, 3, 192, 320, 17, 17, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 1280, 384, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 1, 2048, 384, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(1, 1, 3, 384, 384, 8, 8, 1, (0, 1, 0, 1), 4)
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 2048),
                weight_shape=(1000, 2048),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
        elif self.name == "inception_v3" and self.model_input == (8, 3, 299, 299) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_inception_v3_add_relu(8, 1, 1, 2048, 192, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 1024, 320, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 2048, 448, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 128, 192, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 1280, 320, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 1280, 448, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 3, 1, 384, 384, 8, 8, 1, (1, 0, 1, 0), 4)
            pattern_inception_v3_add_relu(8, 3, 3, 192, 192, 17, 17, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 7, 192, 192, 17, 17, 1, (0, 3, 0, 3), 4)
            pattern_inception_v3_add_relu(8, 7, 1, 192, 192, 17, 17, 1, (3, 0, 3, 0), 4)
            pattern_inception_v3_add_relu(8, 7, 1, 160, 160, 17, 17, 1, (3, 0, 3, 0), 4)
            pattern_inception_v3_add_relu(8, 1, 7, 160, 192, 17, 17, 1, (0, 3, 0, 3), 2)
            pattern_inception_v3_add_relu(8, 1, 1, 768, 160, 17, 17, 1, (0, 0, 0, 0), 4)
            pattern_inception_v3_add_relu(8, 1, 7, 160, 160, 17, 17, 1, (0, 3, 0, 3), 4)
            pattern_inception_v3_add_relu(8, 7, 1, 160, 192, 17, 17, 1, (3, 0, 3, 0), 2)
            pattern_inception_v3_add_relu(8, 7, 1, 128, 128, 17, 17, 1, (3, 0, 3, 0), 2)
            pattern_inception_v3_add_relu(8, 1, 7, 128, 192, 17, 17, 1, (0, 3, 0, 3), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 768, 128, 17, 17, 1, (0, 0, 0, 0), 2)
            pattern_inception_v3_add_relu(8, 1, 7, 128, 128, 17, 17, 1, (0, 3, 0, 3), 2)
            pattern_inception_v3_add_relu(8, 7, 1, 128, 192, 17, 17, 1, (3, 0, 3, 0), 1)
            pattern_inception_v3_add_relu(8, 3, 3, 96, 96, 35, 35, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 288, 48, 35, 35, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 256, 48, 35, 35, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 192, 32, 35, 35, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 192, 48, 35, 35, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 5, 5, 48, 46, 35, 35, 1, (2, 2, 2, 2), 3)
            pattern_inception_v3_add_relu(8, 3, 3, 3, 32, 299, 299, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 3, 3, 32, 32, 149, 149, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 3, 3, 32, 64, 147, 147, 1, (1, 1, 1, 1), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 64, 80, 73, 73, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 192, 64, 35, 35, 1, (0, 0, 0, 0), 2)
            pattern_inception_v3_add_relu(8, 1, 1, 256, 64, 35, 35, 1, (0, 0, 0, 0), 3)
            pattern_inception_v3_add_relu(8, 1, 1, 288, 64, 35, 35, 1, (0, 0, 0, 0,), 4)
            pattern_inception_v3_add_relu(8, 3, 3, 288, 384, 35, 35, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 768, 192, 17, 17, 1, (0, 0, 0, 0), 12)
            pattern_inception_v3_add_relu(8, 3, 3, 192, 320, 17, 17, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 1280, 384, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 1, 2048, 384, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(8, 1, 3, 384, 384, 8, 8, 1, (0, 1, 0, 1), 4)
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 2048),
                weight_shape=(1000, 2048),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
            
            
        elif self.name == "inception_v3" and self.model_input == (16, 3, 299, 299) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_inception_v3_add_relu(16, 1, 1, 2048, 192, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 1024, 320, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 2048, 448, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 128, 192, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 1280, 320, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 1280, 448, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 3, 1, 384, 384, 8, 8, 1, (1, 0, 1, 0), 4)
            pattern_inception_v3_add_relu(16, 3, 3, 192, 192, 17, 17, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 7, 192, 192, 17, 17, 1, (0, 3, 0, 3), 4)
            pattern_inception_v3_add_relu(16, 7, 1, 192, 192, 17, 17, 1, (3, 0, 3, 0), 4)
            pattern_inception_v3_add_relu(16, 7, 1, 160, 160, 17, 17, 1, (3, 0, 3, 0), 4)
            pattern_inception_v3_add_relu(16, 1, 7, 160, 192, 17, 17, 1, (0, 3, 0, 3), 2)
            pattern_inception_v3_add_relu(16, 1, 1, 768, 160, 17, 17, 1, (0, 0, 0, 0), 4)
            pattern_inception_v3_add_relu(16, 1, 7, 160, 160, 17, 17, 1, (0, 3, 0, 3), 4)
            pattern_inception_v3_add_relu(16, 7, 1, 160, 192, 17, 17, 1, (3, 0, 3, 0), 2)
            pattern_inception_v3_add_relu(16, 7, 1, 128, 128, 17, 17, 1, (3, 0, 3, 0), 2)
            pattern_inception_v3_add_relu(16, 1, 7, 128, 192, 17, 17, 1, (0, 3, 0, 3), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 768, 128, 17, 17, 1, (0, 0, 0, 0), 2)
            pattern_inception_v3_add_relu(16, 1, 7, 128, 128, 17, 17, 1, (0, 3, 0, 3), 2)
            pattern_inception_v3_add_relu(16, 7, 1, 128, 192, 17, 17, 1, (3, 0, 3, 0), 1)
            pattern_inception_v3_add_relu(16, 3, 3, 96, 96, 35, 35, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 288, 48, 35, 35, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 256, 48, 35, 35, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 192, 32, 35, 35, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 192, 48, 35, 35, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 5, 5, 48, 46, 35, 35, 1, (2, 2, 2, 2), 3)
            pattern_inception_v3_add_relu(16, 3, 3, 3, 32, 299, 299, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 3, 3, 32, 32, 149, 149, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 3, 3, 32, 64, 147, 147, 1, (1, 1, 1, 1), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 64, 80, 73, 73, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 192, 64, 35, 35, 1, (0, 0, 0, 0), 2)
            pattern_inception_v3_add_relu(16, 1, 1, 256, 64, 35, 35, 1, (0, 0, 0, 0), 3)
            pattern_inception_v3_add_relu(16, 1, 1, 288, 64, 35, 35, 1, (0, 0, 0, 0,), 4)
            pattern_inception_v3_add_relu(16, 3, 3, 288, 384, 35, 35, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 768, 192, 17, 17, 1, (0, 0, 0, 0), 12)
            pattern_inception_v3_add_relu(16, 3, 3, 192, 320, 17, 17, 2, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 1280, 384, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 1, 2048, 384, 8, 8, 1, (0, 0, 0, 0), 1)
            pattern_inception_v3_add_relu(16, 1, 3, 384, 384, 8, 8, 1, (0, 1, 0, 1), 4)
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 2048),
                weight_shape=(1000, 2048),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        # mobilenet_v2
        elif self.name == "mobilenet_v2" and self.model_input == (1, 3, 224, 224) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_mobilenet_add(1, 3, 3, 3, 32, 224, 224, 2, 1, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 32, 1, 112, 112, 1, 1, 32, 1)
            pattern_mobilenet_add(1, 1, 1, 32, 16, 112, 112, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 16, 96, 112, 112, 1, 0, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 96, 1, 112, 112, 2, 1, 96, 1)
            pattern_mobilenet_add(1, 1, 1, 96, 24, 56, 56, 1, 0, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 144, 1, 56, 56, 1, 1, 144, 1)
            pattern_mobilenet_add(1, 1, 1, 144, 24, 56, 56, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 24, 144, 56, 56, 1, 0, 1, 2)
            pattern_mobilenet_add_group(1, 3, 3, 144, 1, 56, 56, 2, 1, 144, 1)
            pattern_mobilenet_add(1, 1, 1, 144, 32, 28, 28, 1, 0, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 192, 1, 28, 28, 1, 1, 192, 2)
            pattern_mobilenet_add(1, 1, 1, 192, 32, 28, 28, 1, 0, 1, 2)
            pattern_mobilenet_add(1, 1, 1, 32, 192, 28, 28, 1, 0, 1, 3)
            pattern_mobilenet_add_group(1, 3, 3, 192, 1, 28, 28, 2, 1, 192, 1)
            pattern_mobilenet_add(1, 1, 1, 192, 64, 14, 14, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 384, 64, 14, 14, 1, 0, 1, 3)
            pattern_mobilenet_add(1, 1, 1, 64, 384, 14, 14, 1, 0, 1, 4)
            pattern_mobilenet_add_group(1, 3, 3, 384, 1, 14, 14, 1, 1, 384, 4)
            pattern_mobilenet_add(1, 1, 1, 384, 96, 14, 14, 1, 0, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 576, 1, 14, 14, 1, 1, 576, 2)
            pattern_mobilenet_add(1, 1, 1, 576, 96, 14, 14, 1, 0, 1, 2)
            pattern_mobilenet_add(1, 1, 1, 96, 576, 14, 14, 1, 0, 1, 3)
            pattern_mobilenet_add_group(1, 3, 3, 576, 1, 14, 14, 2, 1, 576, 1)
            pattern_mobilenet_add(1, 1, 1, 576, 160, 7, 7, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 960, 160, 7, 7, 1, 0, 1, 2)
            pattern_mobilenet_add(1, 1, 1, 160, 960, 7, 7, 1, 0, 1, 3)
            pattern_mobilenet_add_group(1, 3, 3, 960, 1, 7, 7, 1, 1, 960, 3)
            pattern_mobilenet_add(1, 1, 1, 960, 320, 7, 7, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 320, 1280, 7, 7, 1, 0, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 1280),
                weight_shape=(1000, 1280),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
        elif self.name == "mobilenet_v2" and self.model_input == (8, 3, 224, 224) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_mobilenet_add(8, 3, 3, 3, 32, 224, 224, 2, 1, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 32, 1, 112, 112, 1, 1, 32, 1)
            pattern_mobilenet_add(8, 1, 1, 32, 16, 112, 112, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 16, 96, 112, 112, 1, 0, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 96, 1, 112, 112, 2, 1, 96, 1)
            pattern_mobilenet_add(8, 1, 1, 96, 24, 56, 56, 1, 0, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 144, 1, 56, 56, 1, 1, 144, 1)
            pattern_mobilenet_add(8, 1, 1, 144, 24, 56, 56, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 24, 144, 56, 56, 1, 0, 1, 2)
            pattern_mobilenet_add_group(8, 3, 3, 144, 1, 56, 56, 2, 1, 144, 1)
            pattern_mobilenet_add(8, 1, 1, 144, 32, 28, 28, 1, 0, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 192, 1, 28, 28, 1, 1, 192, 2)
            pattern_mobilenet_add(8, 1, 1, 192, 32, 28, 28, 1, 0, 1, 2)
            pattern_mobilenet_add(8, 1, 1, 32, 192, 28, 28, 1, 0, 1, 3)
            pattern_mobilenet_add_group(8, 3, 3, 192, 1, 28, 28, 2, 1, 192, 1)
            pattern_mobilenet_add(8, 1, 1, 192, 64, 14, 14, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 384, 64, 14, 14, 1, 0, 1, 3)
            pattern_mobilenet_add(8, 1, 1, 64, 384, 14, 14, 1, 0, 1, 4)
            pattern_mobilenet_add_group(8, 3, 3, 384, 1, 14, 14, 1, 1, 384, 4)
            pattern_mobilenet_add(8, 1, 1, 384, 96, 14, 14, 1, 0, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 576, 1, 14, 14, 1, 1, 576, 2)
            pattern_mobilenet_add(8, 1, 1, 576, 96, 14, 14, 1, 0, 1, 2)
            pattern_mobilenet_add(8, 1, 1, 96, 576, 14, 14, 1, 0, 1, 3)
            pattern_mobilenet_add_group(8, 3, 3, 576, 1, 14, 14, 2, 1, 576, 1)
            pattern_mobilenet_add(8, 1, 1, 576, 160, 7, 7, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 960, 160, 7, 7, 1, 0, 1, 2)
            pattern_mobilenet_add(8, 1, 1, 160, 960, 7, 7, 1, 0, 1, 3)
            pattern_mobilenet_add_group(8, 3, 3, 960, 1, 7, 7, 1, 1, 960, 3)
            pattern_mobilenet_add(8, 1, 1, 960, 320, 7, 7, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 320, 1280, 7, 7, 1, 0, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 1280),
                weight_shape=(1000, 1280),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "mobilenet_v2" and self.model_input == (16, 3, 224, 224) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_mobilenet_add(16, 3, 3, 3, 32, 224, 224, 2, 1, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 32, 1, 112, 112, 1, 1, 32, 1)
            pattern_mobilenet_add(16, 1, 1, 32, 16, 112, 112, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 16, 96, 112, 112, 1, 0, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 96, 1, 112, 112, 2, 1, 96, 1)
            pattern_mobilenet_add(16, 1, 1, 96, 24, 56, 56, 1, 0, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 144, 1, 56, 56, 1, 1, 144, 1)
            pattern_mobilenet_add(16, 1, 1, 144, 24, 56, 56, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 24, 144, 56, 56, 1, 0, 1, 2)
            pattern_mobilenet_add_group(16, 3, 3, 144, 1, 56, 56, 2, 1, 144, 1)
            pattern_mobilenet_add(16, 1, 1, 144, 32, 28, 28, 1, 0, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 192, 1, 28, 28, 1, 1, 192, 2)
            pattern_mobilenet_add(16, 1, 1, 192, 32, 28, 28, 1, 0, 1, 2)
            pattern_mobilenet_add(16, 1, 1, 32, 192, 28, 28, 1, 0, 1, 3)
            pattern_mobilenet_add_group(16, 3, 3, 192, 1, 28, 28, 2, 1, 192, 1)
            pattern_mobilenet_add(16, 1, 1, 192, 64, 14, 14, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 384, 64, 14, 14, 1, 0, 1, 3)
            pattern_mobilenet_add(16, 1, 1, 64, 384, 14, 14, 1, 0, 1, 4)
            pattern_mobilenet_add_group(16, 3, 3, 384, 1, 14, 14, 1, 1, 384, 4)
            pattern_mobilenet_add(16, 1, 1, 384, 96, 14, 14, 1, 0, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 576, 1, 14, 14, 1, 1, 576, 2)
            pattern_mobilenet_add(16, 1, 1, 576, 96, 14, 14, 1, 0, 1, 2)
            pattern_mobilenet_add(16, 1, 1, 96, 576, 14, 14, 1, 0, 1, 3)
            pattern_mobilenet_add_group(16, 3, 3, 576, 1, 14, 14, 2, 1, 576, 1)
            pattern_mobilenet_add(16, 1, 1, 576, 160, 7, 7, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 960, 160, 7, 7, 1, 0, 1, 2)
            pattern_mobilenet_add(16, 1, 1, 160, 960, 7, 7, 1, 0, 1, 3)
            pattern_mobilenet_add_group(16, 3, 3, 960, 1, 7, 7, 1, 1, 960, 3)
            pattern_mobilenet_add(16, 1, 1, 960, 320, 7, 7, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 320, 1280, 7, 7, 1, 0, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 1280),
                weight_shape=(1000, 1280),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "mobilenet_v2" and self.model_input == (1, 3, 240, 240) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            
            pattern_mobilenet_add(1, 3, 3, 3, 32, 240, 240, 2, 1, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 32, 1, 120, 120, 1, 1, 32, 1)
            pattern_mobilenet_add(1, 1, 1, 32, 16, 120, 120, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 16, 96, 120, 120, 1, 0, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 96, 1, 120, 120, 2, 1, 96, 1)
            pattern_mobilenet_add(1, 1, 1, 96, 24, 60, 60, 1, 0, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 144, 1, 60, 60, 1, 1, 144, 1)
            pattern_mobilenet_add(1, 1, 1, 144, 24, 60, 60, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 24, 144, 60, 60, 1, 0, 1, 2)
            pattern_mobilenet_add_group(1, 3, 3, 144, 1, 60, 60, 2, 1, 144, 1)
            pattern_mobilenet_add(1, 1, 1, 144, 32, 30, 30, 1, 0, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 192, 1, 30, 30, 1, 1, 192, 2)
            pattern_mobilenet_add(1, 1, 1, 192, 32, 30, 30, 1, 0, 1, 2)
            pattern_mobilenet_add(1, 1, 1, 32, 192, 30, 30, 1, 0, 1, 3)
            pattern_mobilenet_add_group(1, 3, 3, 192, 1, 30, 30, 2, 1, 192, 1)
            pattern_mobilenet_add(1, 1, 1, 192, 64, 15, 15, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 384, 64, 15, 15, 1, 0, 1, 3)
            pattern_mobilenet_add(1, 1, 1, 64, 384, 15, 15, 1, 0, 1, 4)
            pattern_mobilenet_add_group(1, 3, 3, 384, 1, 15, 15, 1, 1, 384, 4)
            pattern_mobilenet_add(1, 1, 1, 384, 96, 15, 15, 1, 0, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 576, 1, 15, 15, 1, 1, 576, 2)
            pattern_mobilenet_add(1, 1, 1, 576, 96, 15, 15, 1, 0, 1, 2)
            pattern_mobilenet_add(1, 1, 1, 96, 576, 15, 15, 1, 0, 1, 3)
            pattern_mobilenet_add_group(1, 3, 3, 576, 1, 15, 15, 2, 1, 576, 1)
            pattern_mobilenet_add(1, 1, 1, 576, 160, 8, 8, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 960, 160, 8, 8, 1, 0, 1, 2)
            pattern_mobilenet_add(1, 1, 1, 160, 960, 8, 8, 1, 0, 1, 3)
            pattern_mobilenet_add_group(1, 3, 3, 960, 1, 8, 8, 1, 1, 960, 3)
            pattern_mobilenet_add(1, 1, 1, 960, 320, 8, 8, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 320, 1280, 8, 8, 1, 0, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 1280),
                weight_shape=(1000, 1280),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
            
        elif self.name == "mobilenet_v2" and self.model_input == (8, 3, 240, 240) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_mobilenet_add(8, 3, 3, 3, 32, 240, 240, 2, 1, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 32, 1, 120, 120, 1, 1, 32, 1)
            pattern_mobilenet_add(8, 1, 1, 32, 16, 120, 120, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 16, 96, 120, 120, 1, 0, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 96, 1, 120, 120, 2, 1, 96, 1)
            pattern_mobilenet_add(8, 1, 1, 96, 24, 60, 60, 1, 0, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 144, 1, 60, 60, 1, 1, 144, 1)
            pattern_mobilenet_add(8, 1, 1, 144, 24, 60, 60, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 24, 144, 60, 60, 1, 0, 1, 2)
            pattern_mobilenet_add_group(8, 3, 3, 144, 1, 60, 60, 2, 1, 144, 1)
            pattern_mobilenet_add(8, 1, 1, 144, 32, 30, 30, 1, 0, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 192, 1, 30, 30, 1, 1, 192, 2)
            pattern_mobilenet_add(8, 1, 1, 192, 32, 30, 30, 1, 0, 1, 2)
            pattern_mobilenet_add(8, 1, 1, 32, 192, 30, 30, 1, 0, 1, 3)
            pattern_mobilenet_add_group(8, 3, 3, 192, 1, 30, 30, 2, 1, 192, 1)
            pattern_mobilenet_add(8, 1, 1, 192, 64, 15, 15, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 384, 64, 15, 15, 1, 0, 1, 3)
            pattern_mobilenet_add(8, 1, 1, 64, 384, 15, 15, 1, 0, 1, 4)
            pattern_mobilenet_add_group(8, 3, 3, 384, 1, 15, 15, 1, 1, 384, 4)
            pattern_mobilenet_add(8, 1, 1, 384, 96, 15, 15, 1, 0, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 576, 1, 15, 15, 1, 1, 576, 2)
            pattern_mobilenet_add(8, 1, 1, 576, 96, 15, 15, 1, 0, 1, 2)
            pattern_mobilenet_add(8, 1, 1, 96, 576, 15, 15, 1, 0, 1, 3)
            pattern_mobilenet_add_group(8, 3, 3, 576, 1, 15, 15, 2, 1, 576, 1)
            pattern_mobilenet_add(8, 1, 1, 576, 160, 8, 8, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 960, 160, 8, 8, 1, 0, 1, 2)
            pattern_mobilenet_add(8, 1, 1, 160, 960, 8, 8, 1, 0, 1, 3)
            pattern_mobilenet_add_group(8, 3, 3, 960, 1, 8, 8, 1, 1, 960, 3)
            pattern_mobilenet_add(8, 1, 1, 960, 320, 8, 8, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 320, 1280, 8, 8, 1, 0, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 1280),
                weight_shape=(1000, 1280),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "mobilenet_v2" and self.model_input == (16, 3, 240, 240) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_mobilenet_add(16, 3, 3, 3, 32, 240, 240, 2, 1, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 32, 1, 120, 120, 1, 1, 32, 1)
            pattern_mobilenet_add(16, 1, 1, 32, 16, 120, 120, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 16, 96, 120, 120, 1, 0, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 96, 1, 120, 120, 2, 1, 96, 1)
            pattern_mobilenet_add(16, 1, 1, 96, 24, 60, 60, 1, 0, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 144, 1, 60, 60, 1, 1, 144, 1)
            pattern_mobilenet_add(16, 1, 1, 144, 24, 60, 60, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 24, 144, 60, 60, 1, 0, 1, 2)
            pattern_mobilenet_add_group(16, 3, 3, 144, 1, 60, 60, 2, 1, 144, 1)
            pattern_mobilenet_add(16, 1, 1, 144, 32, 30, 30, 1, 0, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 192, 1, 30, 30, 1, 1, 192, 2)
            pattern_mobilenet_add(16, 1, 1, 192, 32, 30, 30, 1, 0, 1, 2)
            pattern_mobilenet_add(16, 1, 1, 32, 192, 30, 30, 1, 0, 1, 3)
            pattern_mobilenet_add_group(16, 3, 3, 192, 1, 30, 30, 2, 1, 192, 1)
            pattern_mobilenet_add(16, 1, 1, 192, 64, 15, 15, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 384, 64, 15, 15, 1, 0, 1, 3)
            pattern_mobilenet_add(16, 1, 1, 64, 384, 15, 15, 1, 0, 1, 4)
            pattern_mobilenet_add_group(16, 3, 3, 384, 1, 15, 15, 1, 1, 384, 4)
            pattern_mobilenet_add(16, 1, 1, 384, 96, 15, 15, 1, 0, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 576, 1, 15, 15, 1, 1, 576, 2)
            pattern_mobilenet_add(16, 1, 1, 576, 96, 15, 15, 1, 0, 1, 2)
            pattern_mobilenet_add(16, 1, 1, 96, 576, 15, 15, 1, 0, 1, 3)
            pattern_mobilenet_add_group(16, 3, 3, 576, 1, 15, 15, 2, 1, 576, 1)
            pattern_mobilenet_add(16, 1, 1, 576, 160, 8, 8, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 960, 160, 8, 8, 1, 0, 1, 2)
            pattern_mobilenet_add(16, 1, 1, 160, 960, 8, 8, 1, 0, 1, 3)
            pattern_mobilenet_add_group(16, 3, 3, 960, 1, 8, 8, 1, 1, 960, 3)
            pattern_mobilenet_add(16, 1, 1, 960, 320, 8, 8, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 320, 1280, 8, 8, 1, 0, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 1280),
                weight_shape=(1000, 1280),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "mobilenet_v2" and self.model_input == (1, 3, 256, 256) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_mobilenet_add(1, 3, 3, 3, 32, 256, 256, 2, 1, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 32, 1, 128, 128, 1, 1, 32, 1)
            pattern_mobilenet_add(1, 1, 1, 32, 16, 128, 128, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 16, 96, 128, 128, 1, 0, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 96, 1, 128, 128, 2, 1, 96, 1)
            pattern_mobilenet_add(1, 1, 1, 96, 24, 64, 64, 1, 0, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 144, 1, 64, 64, 1, 1, 144, 1)
            pattern_mobilenet_add(1, 1, 1, 144, 24, 64, 64, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 24, 144, 64, 64, 1, 0, 1, 2)
            pattern_mobilenet_add_group(1, 3, 3, 144, 1, 64, 64, 2, 1, 144, 1)
            pattern_mobilenet_add(1, 1, 1, 144, 32, 32, 32, 1, 0, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 192, 1, 32, 32, 1, 1, 192, 2)
            pattern_mobilenet_add(1, 1, 1, 192, 32, 32, 32, 1, 0, 1, 2)
            pattern_mobilenet_add(1, 1, 1, 32, 192, 32, 32, 1, 0, 1, 3)
            pattern_mobilenet_add_group(1, 3, 3, 192, 1, 32, 32, 2, 1, 192, 1)
            pattern_mobilenet_add(1, 1, 1, 192, 64, 16, 16, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 384, 64, 16, 16, 1, 0, 1, 3)
            pattern_mobilenet_add(1, 1, 1, 64, 384, 16, 16, 1, 0, 1, 4)
            pattern_mobilenet_add_group(1, 3, 3, 384, 1, 16, 16, 1, 1, 384, 4)
            pattern_mobilenet_add(1, 1, 1, 384, 96, 16, 16, 1, 0, 1, 1)
            pattern_mobilenet_add_group(1, 3, 3, 576, 1, 16, 16, 1, 1, 576, 2)
            pattern_mobilenet_add(1, 1, 1, 576, 96, 16, 16, 1, 0, 1, 2)
            pattern_mobilenet_add(1, 1, 1, 96, 576, 16, 16, 1, 0, 1, 3)
            pattern_mobilenet_add_group(1, 3, 3, 576, 1, 16, 16, 2, 1, 576, 1)
            pattern_mobilenet_add(1, 1, 1, 576, 160, 8, 8, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 960, 160, 8, 8, 1, 0, 1, 2)
            pattern_mobilenet_add(1, 1, 1, 160, 960, 8, 8, 1, 0, 1, 3)
            pattern_mobilenet_add_group(1, 3, 3, 960, 1, 8, 8, 1, 1, 960, 3)
            pattern_mobilenet_add(1, 1, 1, 960, 320, 8, 8, 1, 0, 1, 1)
            pattern_mobilenet_add(1, 1, 1, 320, 1280, 8, 8, 1, 0, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(1, 1280),
                weight_shape=(1000, 1280),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "mobilenet_v2" and self.model_input == (8, 3, 256, 256) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_mobilenet_add(8, 3, 3, 3, 32, 256, 256, 2, 1, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 32, 1, 128, 128, 1, 1, 32, 1)
            pattern_mobilenet_add(8, 1, 1, 32, 16, 128, 128, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 16, 96, 128, 128, 1, 0, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 96, 1, 128, 128, 2, 1, 96, 1)
            pattern_mobilenet_add(8, 1, 1, 96, 24, 64, 64, 1, 0, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 144, 1, 64, 64, 1, 1, 144, 1)
            pattern_mobilenet_add(8, 1, 1, 144, 24, 64, 64, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 24, 144, 64, 64, 1, 0, 1, 2)
            pattern_mobilenet_add_group(8, 3, 3, 144, 1, 64, 64, 2, 1, 144, 1)
            pattern_mobilenet_add(8, 1, 1, 144, 32, 32, 32, 1, 0, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 192, 1, 32, 32, 1, 1, 192, 2)
            pattern_mobilenet_add(8, 1, 1, 192, 32, 32, 32, 1, 0, 1, 2)
            pattern_mobilenet_add(8, 1, 1, 32, 192, 32, 32, 1, 0, 1, 3)
            pattern_mobilenet_add_group(8, 3, 3, 192, 1, 32, 32, 2, 1, 192, 1)
            pattern_mobilenet_add(8, 1, 1, 192, 64, 16, 16, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 384, 64, 16, 16, 1, 0, 1, 3)
            pattern_mobilenet_add(8, 1, 1, 64, 384, 16, 16, 1, 0, 1, 4)
            pattern_mobilenet_add_group(8, 3, 3, 384, 1, 16, 16, 1, 1, 384, 4)
            pattern_mobilenet_add(8, 1, 1, 384, 96, 16, 16, 1, 0, 1, 1)
            pattern_mobilenet_add_group(8, 3, 3, 576, 1, 16, 16, 1, 1, 576, 2)
            pattern_mobilenet_add(8, 1, 1, 576, 96, 16, 16, 1, 0, 1, 2)
            pattern_mobilenet_add(8, 1, 1, 96, 576, 16, 16, 1, 0, 1, 3)
            pattern_mobilenet_add_group(8, 3, 3, 576, 1, 16, 16, 2, 1, 576, 1)
            pattern_mobilenet_add(8, 1, 1, 576, 160, 8, 8, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 960, 160, 8, 8, 1, 0, 1, 2)
            pattern_mobilenet_add(8, 1, 1, 160, 960, 8, 8, 1, 0, 1, 3)
            pattern_mobilenet_add_group(8, 3, 3, 960, 1, 8, 8, 1, 1, 960, 3)
            pattern_mobilenet_add(8, 1, 1, 960, 320, 8, 8, 1, 0, 1, 1)
            pattern_mobilenet_add(8, 1, 1, 320, 1280, 8, 8, 1, 0, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(8, 1280),
                weight_shape=(1000, 1280),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        elif self.name == "mobilenet_v2" and self.model_input == (16, 3, 256, 256) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_mobilenet_add(16, 3, 3, 3, 32, 256, 256, 2, 1, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 32, 1, 128, 128, 1, 1, 32, 1)
            pattern_mobilenet_add(16, 1, 1, 32, 16, 128, 128, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 16, 96, 128, 128, 1, 0, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 96, 1, 128, 128, 2, 1, 96, 1)
            pattern_mobilenet_add(16, 1, 1, 96, 24, 64, 64, 1, 0, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 144, 1, 64, 64, 1, 1, 144, 1)
            pattern_mobilenet_add(16, 1, 1, 144, 24, 64, 64, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 24, 144, 64, 64, 1, 0, 1, 2)
            pattern_mobilenet_add_group(16, 3, 3, 144, 1, 64, 64, 2, 1, 144, 1)
            pattern_mobilenet_add(16, 1, 1, 144, 32, 32, 32, 1, 0, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 192, 1, 32, 32, 1, 1, 192, 2)
            pattern_mobilenet_add(16, 1, 1, 192, 32, 32, 32, 1, 0, 1, 2)
            pattern_mobilenet_add(16, 1, 1, 32, 192, 32, 32, 1, 0, 1, 3)
            pattern_mobilenet_add_group(16, 3, 3, 192, 1, 32, 32, 2, 1, 192, 1)
            pattern_mobilenet_add(16, 1, 1, 192, 64, 16, 16, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 384, 64, 16, 16, 1, 0, 1, 3)
            pattern_mobilenet_add(16, 1, 1, 64, 384, 16, 16, 1, 0, 1, 4)
            pattern_mobilenet_add_group(16, 3, 3, 384, 1, 16, 16, 1, 1, 384, 4)
            pattern_mobilenet_add(16, 1, 1, 384, 96, 16, 16, 1, 0, 1, 1)
            pattern_mobilenet_add_group(16, 3, 3, 576, 1, 16, 16, 1, 1, 576, 2)
            pattern_mobilenet_add(16, 1, 1, 576, 96, 16, 16, 1, 0, 1, 2)
            pattern_mobilenet_add(16, 1, 1, 96, 576, 16, 16, 1, 0, 1, 3)
            pattern_mobilenet_add_group(16, 3, 3, 576, 1, 16, 16, 2, 1, 576, 1)
            pattern_mobilenet_add(16, 1, 1, 576, 160, 8, 8, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 960, 160, 8, 8, 1, 0, 1, 2)
            pattern_mobilenet_add(16, 1, 1, 160, 960, 8, 8, 1, 0, 1, 3)
            pattern_mobilenet_add_group(16, 3, 3, 960, 1, 8, 8, 1, 1, 960, 3)
            pattern_mobilenet_add(16, 1, 1, 960, 320, 8, 8, 1, 0, 1, 1)
            pattern_mobilenet_add(16, 1, 1, 320, 1280, 8, 8, 1, 0, 1, 1)
            param_dict = fused_nn_dense_add_template(
                input_shape=(16, 1280),
                weight_shape=(1000, 1280),
                add_shape=(1, 1000)
            )
            (rm, p) = self.op_utils.get_mod_params(
                'fused_nn_dense_add', 
                param_dict
            )
            self.mod_params_list.append((rm, p, 1))
        
        # Conv3d
        elif self.name == "resnet3d_18" and self.model_input == (1, 16, 3, 112, 112) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_conv3d(1, 28, 14, 2, 256, 512, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 0, 0)
            pattern_conv3d(1, 56, 28, 4, 128, 256, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(1, 112, 56, 8, 64, 128, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(1, 112, 112, 16, 3, 64, 3, 7, 7, (1, 2, 2), (1, 3, 3, 1, 3, 3), 1, 1, 1)
            pattern_conv3d(1, 112, 56, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(1, 112, 56, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(1, 112, 56, 8, 64, 128, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 56, 28, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 56, 28, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(1, 56, 28, 4, 128, 256, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 28, 14, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 28, 14, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(1, 28, 14, 2, 256, 512, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 14, 7, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 14, 7, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 0)
            pattern_conv3d(1, 14, 7, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 0, 0)
            # param_dict = fused_nn_dense_add_template(
            #     input_shape=(1, 512),
            #     weight_shape=(400, 512),
            #     add_shape=(1, 400)
            # )
            # (rm, p) = self.op_utils.get_mod_params(
            #     'fused_nn_dense_add', 
            #     param_dict
            # )
            # self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "resnet3d_18" and self.model_input == (8, 16, 3, 112, 112) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_conv3d(8, 28, 14, 2, 256, 512, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(8, 56, 28, 4, 128, 256, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(8, 112, 56, 8, 64, 128, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(8, 112, 112, 16, 3, 64, 3, 7, 7, (1, 2, 2), (1, 3, 3, 1, 3, 3), 1, 1, 1)
            pattern_conv3d(8, 112, 56, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(8, 112, 56, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(8, 112, 56, 8, 64, 128, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 56, 28, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 56, 28, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(8, 56, 28, 4, 128, 256, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 28, 14, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 28, 14, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(8, 28, 14, 2, 256, 512, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 14, 7, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 14, 7, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 0)
            pattern_conv3d(8, 14, 7, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 0, 0)
            # param_dict = fused_nn_dense_add_template(
            #     input_shape=(8, 512),
            #     weight_shape=(400, 512),
            #     add_shape=(1, 400)
            # )
            # (rm, p) = self.op_utils.get_mod_params(
            #     'fused_nn_dense_add', 
            #     param_dict
            # )
            # self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "resnet3d_18" and self.model_input == (16, 16, 3, 112, 112) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_conv3d(16, 28, 14, 2, 256, 512, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(16, 56, 28, 4, 128, 256, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(16, 112, 56, 8, 64, 128, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(16, 112, 112, 16, 3, 64, 3, 7, 7, (1, 2, 2), (1, 3, 3, 1, 3, 3), 1, 1, 1)
            pattern_conv3d(16, 112, 56, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(16, 112, 56, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(16, 112, 56, 8, 64, 128, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 56, 28, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 56, 28, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(16, 56, 28, 4, 128, 256, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 28, 14, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 28, 14, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(16, 28, 14, 2, 256, 512, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 14, 7, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 14, 7, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 0)
            pattern_conv3d(16, 14, 7, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 0, 0)
            # param_dict = fused_nn_dense_add_template(
            #     input_shape=(16, 512),
            #     weight_shape=(400, 512),
            #     add_shape=(1, 400)
            # )
            # (rm, p) = self.op_utils.get_mod_params(
            #     'fused_nn_dense_add', 
            #     param_dict
            # )
            # self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "resnet3d_18" and self.model_input == (1, 16, 3, 128, 128) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_conv3d(1, 32, 16, 2, 256, 512, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(1, 64, 32, 4, 128, 256, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(1, 128, 64, 8, 64, 128, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(1, 128, 128, 16, 3, 64, 3, 7, 7, (1, 2, 2), (1, 3, 3, 1, 3, 3), 1, 1, 1)
            pattern_conv3d(1, 128, 64, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(1, 128, 64, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(1, 128, 64, 8, 64, 128, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 64, 32, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 64, 32, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(1, 64, 32, 4, 128, 256, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 32, 16, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 32, 16, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(1, 32, 16, 2, 256, 512, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 16, 8, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 16, 8, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 0)
            pattern_conv3d(1, 16, 8, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 0, 0)
            # param_dict = fused_nn_dense_add_template(
            #     input_shape=(1, 512),
            #     weight_shape=(400, 512),
            #     add_shape=(1, 400)
            # )
            # (rm, p) = self.op_utils.get_mod_params(
            #     'fused_nn_dense_add', 
            #     param_dict
            # )
            # self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "resnet3d_18" and self.model_input == (1, 16, 3, 128, 128) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_conv3d(1, 32, 16, 2, 256, 512, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(1, 64, 32, 4, 128, 256, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(1, 128, 64, 8, 64, 128, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(1, 128, 128, 16, 3, 64, 3, 7, 7, (1, 2, 2), (1, 3, 3, 1, 3, 3), 1, 1, 1)
            pattern_conv3d(1, 128, 64, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(1, 128, 64, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(1, 128, 64, 8, 64, 128, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 64, 32, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 64, 32, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(1, 64, 32, 4, 128, 256, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 32, 16, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 32, 16, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(1, 32, 16, 2, 256, 512, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 16, 8, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 16, 8, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 0)
            pattern_conv3d(1, 16, 8, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 0, 0)
            # param_dict = fused_nn_dense_add_template(
            #     input_shape=(1, 512),
            #     weight_shape=(400, 512),
            #     add_shape=(1, 400)
            # )
            # (rm, p) = self.op_utils.get_mod_params(
            #     'fused_nn_dense_add', 
            #     param_dict
            # )
            # self.mod_params_list.append((rm, p, 1))
        elif self.name == "resnet3d_18" and self.model_input == (8, 16, 3, 128, 128) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_conv3d(8, 32, 16, 2, 256, 512, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(8, 64, 32, 4, 128, 256, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(8, 128, 64, 8, 64, 128, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(8, 128, 128, 16, 3, 64, 3, 7, 7, (1, 2, 2), (1, 3, 3, 1, 3, 3), 1, 1, 1)
            pattern_conv3d(8, 128, 64, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(8, 128, 64, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(8, 128, 64, 8, 64, 128, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 64, 32, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 64, 32, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(8, 64, 32, 4, 128, 256, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 32, 16, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 32, 16, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(8, 32, 16, 2, 256, 512, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 16, 8, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 16, 8, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 0)
            pattern_conv3d(8, 16, 8, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 0, 0)
            # param_dict = fused_nn_dense_add_template(
            #     input_shape=(8, 512),
            #     weight_shape=(400, 512),
            #     add_shape=(1, 400)
            # )
            # (rm, p) = self.op_utils.get_mod_params(
            #     'fused_nn_dense_add', 
            #     param_dict
            # )
            # self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "resnet3d_18" and self.model_input == (16, 16, 3, 128, 128) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_conv3d(16, 32, 16, 2, 256, 512, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(16, 64, 32, 4, 128, 256, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(16, 128, 64, 8, 64, 128, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(16, 128, 128, 16, 3, 64, 3, 7, 7, (1, 2, 2), (1, 3, 3, 1, 3, 3), 1, 1, 1)
            pattern_conv3d(16, 128, 64, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(16, 128, 64, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(16, 128, 64, 8, 64, 128, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 64, 32, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 64, 32, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(16, 64, 32, 4, 128, 256, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 32, 16, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 32, 16, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(16, 32, 16, 2, 256, 512, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 16, 8, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 16, 8, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 0)
            pattern_conv3d(16, 16, 8, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 0, 0)
            # param_dict = fused_nn_dense_add_template(
            #     input_shape=(16, 512),
            #     weight_shape=(400, 512),
            #     add_shape=(1, 400)
            # )
            # (rm, p) = self.op_utils.get_mod_params(
            #     'fused_nn_dense_add', 
            #     param_dict
            # )
            # self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "resnet3d_18" and self.model_input == (1, 16, 3, 144, 144) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_conv3d(1, 36, 18, 2, 256, 512, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(1, 72, 36, 4, 128, 256, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(1, 144, 72, 8, 64, 128, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(1, 144, 144, 16, 3, 64, 3, 7, 7, (1, 2, 2), (1, 3, 3, 1, 3, 3), 1, 1, 1)
            pattern_conv3d(1, 144, 72, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(1, 144, 72, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(1, 144, 72, 8, 64, 128, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 72, 36, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 72, 36, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(1, 72, 36, 4, 128, 256, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 36, 18, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 36, 18, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(1, 36, 18, 2, 256, 512, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 18, 9, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(1, 18, 9, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 0)
            pattern_conv3d(1, 18, 9, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 0, 0)
            # param_dict = fused_nn_dense_add_template(
            #     input_shape=(1, 512),
            #     weight_shape=(400, 512),
            #     add_shape=(1, 400)
            # )
            # (rm, p) = self.op_utils.get_mod_params(
            #     'fused_nn_dense_add', 
            #     param_dict
            # )
            # self.mod_params_list.append((rm, p, 1))
        
        elif self.name == "resnet3d_18" and self.model_input == (8, 16, 3, 144, 144) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_conv3d(8, 36, 18, 2, 256, 512, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(8, 72, 36, 4, 128, 256, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(8, 144, 72, 8, 64, 128, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(8, 144, 144, 16, 3, 64, 3, 7, 7, (1, 2, 2), (1, 3, 3, 1, 3, 3), 1, 1, 1)
            pattern_conv3d(8, 144, 72, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(8, 144, 72, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(8, 144, 72, 8, 64, 128, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 72, 36, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 72, 36, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(8, 72, 36, 4, 128, 256, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 36, 18, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 36, 18, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(8, 36, 18, 2, 256, 512, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 18, 9, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(8, 18, 9, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 0)
            pattern_conv3d(8, 18, 9, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 0, 0)
            # param_dict = fused_nn_dense_add_template(
            #     input_shape=(8, 512),
            #     weight_shape=(400, 512),
            #     add_shape=(1, 400)
            # )
            # (rm, p) = self.op_utils.get_mod_params(
            #     'fused_nn_dense_add', 
            #     param_dict
            # )
            # self.mod_params_list.append((rm, p, 1))
        elif self.name == "resnet3d_18" and self.model_input == (16, 16, 3, 144, 144) and \
            self.model_dtype == "int8" and self.model_layout == "NCHW":
            pattern_conv3d(16, 36, 18, 2, 256, 512, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(16, 72, 36, 4, 128, 256, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(16, 144, 72, 8, 64, 128, 1, 1, 1, (2, 2, 2), (0, 0, 0, 0, 0, 0), 1, 1, 0)
            pattern_conv3d(16, 144, 144, 16, 3, 64, 3, 7, 7, (1, 2, 2), (1, 3, 3, 1, 3, 3), 1, 1, 1)
            pattern_conv3d(16, 144, 72, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 1, 1)
            pattern_conv3d(16, 144, 72, 8, 64, 64, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(16, 144, 72, 8, 64, 128, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 72, 36, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 72, 36, 4, 128, 128, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(16, 72, 36, 4, 128, 256, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 36, 18, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 36, 18, 2, 256, 256, 3, 3, 3, (1, 1, 1), (1, 1, 1, 1, 1, 1), 2, 0, 0)
            pattern_conv3d(16, 36, 18, 2, 256, 512, 3, 3, 3, (2, 2, 2), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 18, 9, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 1)
            pattern_conv3d(16, 18, 9, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 1, 0)
            pattern_conv3d(16, 18, 9, 1, 512, 512, 3, 3, 3,  (1, 1, 1), (1, 1, 1, 1, 1, 1), 1, 0, 0)
            # param_dict = fused_nn_dense_add_template(
            #     input_shape=(16, 512),
            #     weight_shape=(400, 512),
            #     add_shape=(1, 400)
            # )
            # (rm, p) = self.op_utils.get_mod_params(
            #     'fused_nn_dense_add', 
            #     param_dict
            # )
            # self.mod_params_list.append((rm, p, 1))
        
        else:
            raise ValueError(f"no support for {self.name} with {self.model_input}")
    
    
    
    
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.collect()
        return self.mod_params_list