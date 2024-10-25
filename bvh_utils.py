from morton_code import *
from aabb import *

@wp.struct
class Node:
    parent_idx: int
    left_idx: int
    right_idx: int
    object_idx: int

@wp.struct
class BasicDeviceBVH:
    num_nodes: int   # number of internal nodes + leaves
    num_objects: int  # number of objects (same as leaves)

    # Nodes, AABBs, and objects are represented as arrays on the device
    nodes: wp.array(dtype=Node)  # array of nodes
    aabbs: wp.array(dtype=aabb)  # array of AABBs
    objects: wp.array(dtype=wpuint32)  # array of object indices


@wp.func
def determine_range(node_code: wp.array(dtype=wpuint64), num_leaves: int, idx: int) -> pair:
    # If idx == 0, return full range
    if idx == 0:
        return pair(wpuint32(0), wpuint32(num_leaves - 1))

    # Determine direction of the range
    self_code = node_code[idx]
    L_delta = common_upper_bits(self_code, node_code[idx - 1])
    R_delta = common_upper_bits(self_code, node_code[idx + 1])

    d = int(-1)

    if R_delta > L_delta:
        d = int(1)

    # Compute upper bound for the length of the range
    delta_min = min(L_delta, R_delta)
    l_max = int(2)
    delta = -1
    i_tmp = idx + d * l_max

    # Expand range until the delta becomes less than delta_min
    if 0 <= i_tmp < num_leaves:
        delta = common_upper_bits(self_code, node_code[i_tmp])

    while delta > delta_min:
        l_max <<= 1
        i_tmp = idx + d * l_max
        delta = -1
        if 0 <= i_tmp < num_leaves:
            delta = common_upper_bits(self_code, node_code[i_tmp])

    # Binary search to find the exact end of the range
    l = int(0)
    t = int(l_max >> 1)
    while t > 0:
        i_tmp = idx + (l + t) * d
        delta = -1
        if 0 <= i_tmp < num_leaves:
            delta = common_upper_bits(self_code, node_code[i_tmp])
        if delta > delta_min:
            l += t
        t >>= 1

    # Compute the final jdx
    jdx = idx + l * d

    # Ensure idx < jdx by swapping if necessary
    if d < 0:
        idx, jdx = jdx, idx

    return pair(wpuint32(idx), wpuint32(jdx))


@wp.func
def find_split(node_code: wp.array(dtype=wpuint64), first: wpuint32, last: wpuint32) -> wpuint32:
    # Get Morton codes of the first and last nodes
    first_code = node_code[first]
    last_code = node_code[last]

    # If first and last codes are the same, return the midpoint
    if first_code == last_code:
        return (first + last) >> wpuint32(1)

    # Calculate delta between first and last codes
    delta_node = common_upper_bits(first_code, last_code)

    # Binary search for the split point
    split = first
    stride = last - first

    while True:
        stride = (stride + wpuint32(1)) >> wpuint32(1)
        middle = split + stride

        if middle < last:
            delta = common_upper_bits(first_code, node_code[middle])
            if delta > delta_node:
                split = middle

        if stride <= 1:
            break

    return split

@wp.kernel
def construct_internal_nodes_kernel(nodes: wp.array(dtype=Node),  # node array
                                    node_code: wp.array(dtype=wp.uint64),  # Morton codes array
                                    num_objects: int):
    idx = wp.tid()  # Get thread id, equivalent to `idx` in the CUDA lambda

    if idx >= num_objects - 1:
        return

    # Set the node as an internal node (object_idx == 0xFFFFFFFF)
    nodes[idx].object_idx = 0xFFFFFFFF

    # Determine the range and find the split
    ij = determine_range(node_code, num_objects, idx)
    gamma = find_split(node_code, ij[0], ij[1])

    # Set left and right child nodes
    nodes[idx].left_idx = int(gamma)
    nodes[idx].right_idx = int(gamma) + 1

    if min(ij[0], ij[1]) == gamma:
        nodes[idx].left_idx += num_objects - 1
    if max(ij[0], ij[1]) == gamma + wpuint32(1):
        nodes[idx].right_idx += num_objects - 1

    # Set parent index for the left and right child nodes
    nodes[nodes[idx].left_idx].parent_idx = idx
    nodes[nodes[idx].right_idx].parent_idx = idx

@wp.func
def calculate(box: aabb, whole: aabb) -> wpuint32:
    p = centroid(box)

    # Normalize the point relative to the whole AABB
    p.x -= whole[1, 0]  # subtract lower.x
    p.y -= whole[1, 2]  # subtract lower.y
    p.z -= whole[1, 2]  # subtract lower.z

    p.x /= (whole[0, 0] - whole[1, 0])  # normalize based on AABB width
    p.y /= (whole[0, 1] - whole[1, 1])  # normalize based on AABB height
    p.z /= (whole[0, 2] - whole[1, 2])  # normalize based on AABB depth

    # Return the Morton code for the normalized point
    return wpuint32(morton_code(p))

@wp.kernel
def transform_aabb_morton_kernel(
        AABBs: wp.array(dtype=aabb),
        mortons: wp.array(dtype=wpuint32),
        whole: aabb
):
    idx = wp.tid()

    mortons[idx] = calculate(AABBs[idx], whole)
#
# @wp.kernel
# def transform_morton64_kernel(
#         mortons: wp.array(dtype=wpuint32),
#         indices: wp.array(dtype=wpuint32),
#         mortons64: wp.array(dtype=wpuint64)
# ):
#     idx = wp.tid()
#
#     mortons64[idx] = wpuint64(mortons[idx]) << wpuint64(32) | wpuint64(indices[idx])
#
# @wp.kernel
# def init_nodes_kernel(
#         nodes: wp.array(dtype=Node),
#         offset: int,
#         indices: wp.array(dtype=wpuint32)
# ):
#     idx = wp.tid()
#     nodes[idx].left_idx = 0xFFFFFFFF
#     nodes[idx].right_idx = 0xFFFFFFFF
#     nodes[idx].parent_idx = 0xFFFFFFFF
#     if idx >= offset:
#         nodes[idx].object_idx = int(indices[idx - offset])
#     else :
#         nodes[idx].object_idx = 0xFFFFFFFF
#
# @wp.kernel
# def init_bvh_kernel(
#         flags: wp.array(dtype=wpuint32),
#         nodes: wp.array(dtype=Node),
#         aabbs: wp.array(dtype=aabb),
#         offset: int
# ):
#     idx = wp.tid() + offset
#
#     parent = nodes[idx].parent_idx
#     while parent != 0xFFFFFFFF:
#         flag = wp.atomic_add(flags, parent, wpuint32(1))
#         if flag == 0:
#             break
#         else :
#             lidx = nodes[parent].left_idx
#             ridx = nodes[parent].right_idx
#             aabbs[parent] = merge(aabbs[lidx], aabbs[ridx])
#             parent = nodes[parent].parent_idx
#
# # query object indices that potentially overlaps with query aabb.
# @wp.func
# def query_overlap(
#         AABBs: wp.array(dtype=aabb),
#         nodes: wp.array(dtype=Node),
#         ansbuffer: wp.array(dtype=wpuint32),
#         stackbuffer: wp.array(dtype=wpuint32),
#         q_aabb: aabb,
#         offset: int,
#         count: int,
#         stack_offset: int
# ) -> int:
#     start = stack_offset
#     stackbuffer[stack_offset] = wpuint32(0)
#     stack_offset += 1
#     num_found = int(0)
#
#     while stack_offset > start and num_found < count:
#         idx = stackbuffer[stack_offset]
#         stack_offset -= 1
#
#         Lidx = nodes[idx].left_idx
#         Ridx = nodes[idx].right_idx
#         Oidx = nodes[idx].object_idx
#
#         if Oidx != 0xFFFFFFFF:
#             ansbuffer[offset + num_found] = wpuint32(Oidx)
#             num_found += 1
#
#         if Lidx != 0xFFFFFFFF and intersects(q_aabb, AABBs[Lidx]):
#             stackbuffer[stack_offset] = wpuint32(Lidx)
#             stack_offset += 1
#
#         if Ridx != 0xFFFFFFFF and intersects(q_aabb, AABBs[Ridx]):
#             stackbuffer[stack_offset] = wpuint32(Ridx)
#             stack_offset += 1
#
#     return num_found
#
# @wp.kernel
# def query_overlap_kernel(
#         AABBs: wp.array(dtype=aabb),
#         nodes: wp.array(dtype=Node),
#         ansbuffer: wp.array(dtype=wpuint32),
#         num_found: wp.array(dtype=wpuint32),
#         stackbuffer: wp.array(dtype=wpuint32),
#         q_aabbs: wp.array(dtype=aabb),
#         max_ans_count: int=64,
#         max_stack_size: int=64,
# ) -> int:
#     idx = wp.tid()
#
#     num_found[idx] = wpuint32(query_overlap(
#         AABBs,
#         nodes,
#         ansbuffer,
#         stackbuffer,
#         q_aabbs[idx],
#         idx * max_ans_count,
#         max_ans_count,
#         idx * max_stack_size
#     ))