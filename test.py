import torch
import warp as wp
import time

wp.init()
from bvh import *



# torch.random.manual_seed(0)

N = 1000000

pts = torch.rand(N, 3) * 500

box = torch.stack((pts + 5, pts - 5), dim=-1).permute(0, 2, 1).contiguous().cuda()
qbox = torch.stack((pts, pts), dim=-1).permute(0, 2, 1).contiguous().cuda()


bvh_test_ = bvh(box)
bvh_test_.contruct()
bvh_test = bvh(box)
t0 = time.time()
bvh_test.contruct()
t1 = time.time()
print("construct bvh", f"{t1 - t0:.6f}")

ansbuffer = wp.zeros(shape=N*64, dtype=wpuint32)
num_found = wp.zeros(shape=N, dtype=wpuint32)
stackbuffer = wp.zeros(shape=N*64, dtype=wpuint32)
t0 = time.time()
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
        32,
        32
    ]
)
wp.synchronize()
t1 = time.time()
print("query bvh", f"{t1 - t0:.6f}")

# print(box)
# print(qbox)
print(num_found)
