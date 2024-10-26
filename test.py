import torch
import warp as wp

wp.init()
from bvh import *



# torch.random.manual_seed(0)

N = 1000000

pts = torch.rand(N, 3) * 400

box = torch.stack((pts + 5, pts - 5), dim=-1).permute(0, 2, 1).contiguous().cuda()
qbox = torch.stack((pts, pts), dim=-1).permute(0, 2, 1).contiguous().cuda()

bvh_test = bvh(box)

bvh_test.contruct()

ansbuffer = wp.zeros(shape=N*64, dtype=wpuint32)
num_found = wp.zeros(shape=N, dtype=wpuint32)
stackbuffer = wp.zeros(shape=N*64, dtype=wpuint32)

wp.launch(
    kernel=query_overlap_kernel,
    dim=(N,),
    inputs=[
        bvh_test.AABBs,
        bvh_test.nodes,
        ansbuffer,
        num_found,
        stackbuffer,
        wp.from_torch(qbox, dtype=aabb),
        64,
        64
    ]
)
wp.synchronize()

print(box)
print(qbox)
print(num_found)
