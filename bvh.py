from numpy.core.numeric import infty
from sympy import shape

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
        return pair(0, num_leaves - 1)

    # Determine direction of the range
    self_code = node_code[idx]
    L_delta = common_upper_bits(self_code, node_code[idx - 1])
    R_delta = common_upper_bits(self_code, node_code[idx + 1])
    d = 1 if R_delta > L_delta else -1

    # Compute upper bound for the length of the range
    delta_min = min(L_delta, R_delta)
    l_max = 2
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
    l = 0
    t = l_max >> 1
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

    return pair(idx, jdx)


@wp.func
def find_split(node_code: wp.array(dtype=wpuint64), num_leaves: wpuint32, first: wpuint32, last: wpuint32) -> wpuint32:
    # Get Morton codes of the first and last nodes
    first_code = node_code[first]
    last_code = node_code[last]

    # If first and last codes are the same, return the midpoint
    if first_code == last_code:
        return (first + last) >> 1

    # Calculate delta between first and last codes
    delta_node = common_upper_bits(first_code, last_code)

    # Binary search for the split point
    split = first
    stride = last - first

    while True:
        stride = (stride + 1) >> 1
        middle = split + stride

        if middle < last:
            delta = common_upper_bits(first_code, node_code[middle])
            if delta > delta_node:
                split = middle

        if stride <= 1:
            break

    return split

@wp.kernel
def construct_internal_nodes_kernel(nodes: wp.array(dtype=wp.uint32),  # node array
                                    node_code: wp.array(dtype=wp.uint64),  # Morton codes array
                                    num_objects: int):
    idx = wp.tid()  # Get thread id, equivalent to `idx` in the CUDA lambda

    if idx >= num_objects - 1:
        return

    # Set the node as an internal node (object_idx == 0xFFFFFFFF)
    nodes[idx].object_idx = 0xFFFFFFFF

    # Determine the range and find the split
    ij = determine_range(node_code, num_objects, idx)
    gamma = find_split(node_code, num_objects, ij[0], ij[1])

    # Set left and right child nodes
    nodes[idx].left_idx = gamma
    nodes[idx].right_idx = gamma + 1

    if min(ij[0], ij[1]) == gamma:
        nodes[idx].left_idx += num_objects - 1
    if max(ij[0], ij[1]) == gamma + 1:
        nodes[idx].right_idx += num_objects - 1

    # Set parent index for the left and right child nodes
    nodes[nodes[idx].left_idx].parent_idx = idx
    nodes[nodes[idx].right_idx].parent_idx = idx

@wp.func
def calculate(box: aabb, whole: aabb) -> wpuint32:
    p = centroid(box)

    # Normalize the point relative to the whole AABB
    p.x -= whole.get_row(1).x  # subtract lower.x
    p.y -= whole.get_row(1).y  # subtract lower.y
    p.z -= whole.get_row(1).z  # subtract lower.z

    p.x /= (whole.get_row(0).x - whole.get_row(1).x)  # normalize based on AABB width
    p.y /= (whole.get_row(0).y - whole.get_row(1).y)  # normalize based on AABB height
    p.z /= (whole.get_row(0).z - whole.get_row(1).z)  # normalize based on AABB depth

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

@wp.kernel
def transform_morton64_kernel(
        mortons: wp.array(dtype=wpuint32),
        indices: wp.array(dtype=wpuint32),
        mortons64: wp.array(dtype=wpuint64)
):
    idx = wp.tid()

    mortons64[idx] = mortons[idx] << 32 | indices[idx]

@wp.kernel
def init_nodes_kernel(
        nodes: wp.array(dtype=Node),
        offset: int,
        indices: wp.array(dtype=wpuint32)
):
    idx = wp.tid()
    nodes[idx].left_idx = 0xFFFFFFFF
    nodes[idx].right_idx = 0xFFFFFFFF
    nodes[idx].parent_idx = 0xFFFFFFFF
    if idx >= offset:
        nodes[idx].object_idx = indices[idx - offset]
    else :
        nodes[idx].object_idx = 0xFFFFFFFF

@wp.kernel
def init_bvh_kernel(
        flags: wp.array(dtype=wpuint32),
        nodes: wp.array(dtype=Node),
        aabbs: wp.array(dtype=aabb),
        offset: int
):
    idx = wp.tid() + offset

    parent = nodes[idx].parent_idx
    while parent != 0xFFFFFFFF:
        flag = wp.atomic_add(flags, parent, 1)
        if flag == 0:
            break
        else :
            lidx = nodes[parent].left_idx
            ridx = nodes[parent].right_idx
            aabbs[parent] = merge(aabbs[lidx], aabbs[ridx])
            parent = nodes[parent].parent_idx

# query object indices that potentially overlaps with query aabb.
@wp.func
def query_overlap(
        AABBs: wp.array(dtype=aabb),
        nodes: wp.array(dtype=Node),
        ansbuffer: wp.array(dtype=wpuint32),
        stackbuffer: wp.array(dtype=wpuint32),
        q_aabb: aabb,
        offset: int,
        count: int,
        stack_offset: int
) -> int:
    start = stack_offset
    stackbuffer[stack_offset] = 0
    stack_offset += 1
    num_found = 0

    while stack_offset > start and num_found < count:
        idx = stackbuffer[stack_offset]
        stack_offset -= 1

        Lidx = nodes[idx].left_idx
        Ridx = nodes[idx].right_idx
        Oidx = nodes[idx].object_idx

        if Oidx != 0xFFFFFFFF:
            ansbuffer[offset + num_found] = Oidx
            num_found += 1

        if Lidx != 0xFFFFFFFF and intersects(q_aabb, AABBs[Lidx]):
            stackbuffer[stack_offset] = Lidx
            stack_offset += 1

        if Ridx != 0xFFFFFFFF and intersects(q_aabb, AABBs[Ridx]):
            stackbuffer[stack_offset] = Ridx
            stack_offset += 1

    return num_found

# Wrapper function to launch the kernel
def construct_internal_nodes(self, node_code, num_objects):
    wp.launch(
        kernel=construct_internal_nodes_kernel,
        dim=(num_objects-1,),  # Launch threads for internal nodes
        inputs=[self.nodes, node_code, num_objects]
    )
    wp.synchronize()

class bvh:
    def __init__(self, objects: torch.tensor):
        self.objects = objects
        self.AABBs = torch.emprt(0)
        self.nodes = torch.empty(0)


    def contruct(self):

        if self.objects.shape[0] == 0:
            return

        num_objects = self.objects.shape[0]
        num_internal_nodes = num_objects - 1
        num_nodes = num_objects * 2 - 1

        self.AABBs = wp.zeros(shape=num_nodes, dtype=aabb)
        AABBs_t = self.AABBs.to_torch(dtype=torch.float32)

        default_aabb = aabb(0.0)
        default_aabb.set_row(0, vec3(-infty))
        default_aabb.set_row(1, vec3(infty))

        AABBs_t[num_internal_nodes:] = self.objects

        aabb_whole = torch.stack(
            (
                torch.max(AABBs_t[:, 0, :], dim=0).values,
                torch.min(AABBs_t[:, 1, :], dim=0).values
            )
        )

        mortons = wp.zeros(shape=num_objects, dtype=wpuint32)

        wp.launch(
            kernel=transform_aabb_morton_kernel,
            dim=(num_objects,),
            inputs=[
                wp.from_torch(self.objects, dtype=aabb),
                mortons,
                aabb_whole,
            ]
        )
        wp.synchronize()

        mortons_t = wp.to_torch(mortons, dtype=torch.uint32)

        mortons_sort_t, indices_t = torch.sort(mortons_t)

        AABBs_t[num_internal_nodes:] = AABBs_t[num_internal_nodes:][indices]

        mortons64 = wp.zeros(shape=num_objects, dtype=wpuint64)

        indices = wp.from_torch(indices_t, dtype=wpuint32)

        wp.launch(
            kernel=transform_morton64_kernel,
            dim=(num_objects,),
            inputs=[
                mortons,
                indices,
                mortons64
            ]
        )
        wp.synchronize()

        default_node = Node()
        default_node.left_idx = 0xFFFFFFFF
        default_node.right_idx = 0xFFFFFFFF
        default_node.parent_idx = 0xFFFFFFFF
        default_node.object_idx = 0xFFFFFFFF

        self.nodes = wp.zeros(shape=num_nodes, dtype=Node)

        wp.launch(
            kernel=init_nodes_kernel,
            dim=(num_nodes,),
            inputs=[
                self.nodes,
                num_internal_nodes,
                indices
            ]
        )
        wp.synchronize()

        construct_internal_nodes(self, mortons64, num_objects)

        wp.launch(
            kernel=init_bvh_kernel,
            dim=(num_objects,),
            inputs=[
                wp.zeros(shape=num_nodes, dtype=wpuint32),
                self.nodes,
                self.AABBs,
                num_internal_nodes
            ]
        )



