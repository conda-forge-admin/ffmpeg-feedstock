#!/usr/bin/env python3
"""Regenerate tiny_openvino_model.xml, the OpenVINO smoke-test model.

ffmpeg resolves the OpenVINO C API with dlopen (see recipe/build.sh). Loading
the library and reading a model only proves the loader works; it does not prove
that inference actually runs through the resolved function pointers. This model
exists so the test suite can push a real frame through the openvino DNN backend.

It is deliberately:

  * weight-free -- "out = in + in" needs no Const, so the IR has no .bin
    companion and the whole model is one reviewable text file;
  * not the identity -- doubling the samples means a working run is
    observably different from a frame that merely passed through;
  * statically shaped -- ffmpeg calls ov_const_port_get_shape() on the input
    port, which cannot resolve a dynamic dimension, so H and W must be fixed
    (callers put an explicit scale filter in front, as vf_dnn_processing
    intends);
  * NCHW -- dnn_backend_openvino.c parses output dims as NCHW unconditionally,
    so an NHWC model yields transposed geometry (a 64x64 frame comes out 3x64).

Requires the `openvino` Python package; it is NOT needed to build or test
ffmpeg, only to regenerate this file:

    conda create -n ov -c conda-forge python openvino
    conda run -n ov python generate_tiny_openvino_model.py
"""

import numpy as np
import openvino as ov
from openvino import opset13 as ops

SIZE = 64

parameter = ops.parameter([1, 3, SIZE, SIZE], dtype=np.float32, name="input")
result = ops.result(ops.add(parameter, parameter))
result.get_output_tensor(0).set_names({"output"})

model = ov.Model([result], [parameter], "ffmpeg_openvino_smoketest")
ov.save_model(model, "tiny_openvino_model.xml", compress_to_fp16=False)

# save_model always writes a .bin; ours is empty because the graph has no
# constants, and ov_core_read_model() is happy without it.
import os
if os.path.exists("tiny_openvino_model.bin") and \
        os.path.getsize("tiny_openvino_model.bin") == 0:
    os.remove("tiny_openvino_model.bin")
