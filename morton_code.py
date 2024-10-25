from common import *

@wp.func
def expand_bits(v: wpuint32) -> wpuint32:
    v = (v * wpuint32(0x00010001)) & wpuint32(0xFF0000FF)
    v = (v * wpuint32(0x00000101)) & wpuint32(0x0F00F00F)
    v = (v * wpuint32(0x00000011)) & wpuint32(0xC30C30C3)
    v = (v * wpuint32(0x00000005)) & wpuint32(0x49249249)
    return v

@wp.func
def morton_code(xyz: vec3) -> wpuint32:
    resolution = float(1024.0)
    # Clamp x, y, z coordinates to [0, resolution - 1]
    xyz.x = wp.min(wp.max(xyz.x * resolution, 0.0), resolution - 1.0)
    xyz.y = wp.min(wp.max(xyz.y * resolution, 0.0), resolution - 1.0)
    xyz.z = wp.min(wp.max(xyz.z * resolution, 0.0), resolution - 1.0)

    # Expand bits for x, y, z
    xx = expand_bits(wpuint32(xyz.x))
    yy = expand_bits(wpuint32(xyz.y))
    zz = expand_bits(wpuint32(xyz.z))

    # Combine the bits into a Morton code
    return xx * wpuint32(4) + yy * wpuint32(2) + zz

@wp.func
def clz(x: wpuint32) -> int:
    if x == 0:
        return 32
    n = 0
    while (x & 0x80000000) == 0:
        x <<= 1
        n += 1
    return n

@wp.func
def clzll(x: wpuint64) -> int:
    if x == 0:
        return 64
    n = int(0)
    while (x & wpuint64(0x8000000000000000)) == 0:
        x <<= wpuint64(1)
        n += 1
    return n

# @wp.func
# def common_upper_bits(lhs: wpuint32, rhs: wpuint32) -> int:
#     # XOR lhs and rhs, and then count leading zeros
#     diff = lhs ^ rhs
#     return clz(diff)


@wp.func
def common_upper_bits(lhs: wpuint64, rhs: wpuint64) -> int:
    # XOR lhs and rhs, and then count leading zeros
    diff = lhs ^ rhs
    return clzll(diff)
