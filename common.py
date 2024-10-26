import torch
import numpy as np
import warp as wp

wpfloat = wp.float32
wpuint32 = wp.uint32
wpuint64 = wp.uint64
vec4 = wp.vec(length=4, dtype=wpfloat)
vec3 = wp.vec(length=3, dtype=wpfloat)
pair = wp.vec(length=2, dtype=wpuint32)

infty = 1e12
Err = wpuint32(0xFFFFFFFF)