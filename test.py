import torch
import warp as wp
import time

wp.init()
from bvh import *



# torch.random.manual_seed(0)

N = 3000

pts = torch.rand(N, 3) * 50

box = torch.stack((pts - 5, pts + 5), dim=-1).permute(0, 2, 1).contiguous().cuda()
qbox = torch.stack((pts, pts), dim=-1).permute(0, 2, 1).contiguous().cuda()


bvh_test_ = bvh(box)
bvh_test_.construct()
bvh_test = bvh(box)
t0 = time.time()
bvh_test.construct()
t1 = time.time()
print("construct bvh", f"{t1 - t0:.6f}")

ansbuffer = wp.zeros(shape=N*64, dtype=wpuint32)
num_found = wp.zeros(shape=N, dtype=wp.int32)
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
        64,
        64
    ]
)
wp.synchronize()
t1 = time.time()
print("query bvh", f"{t1 - t0:.6f}")

# print(box)
# print(qbox)
print(num_found)

def is_overlap(box1, box2):
    return (
        (box1[0, 0] >= box2[0, 0] and box1[1, 0] <= box2[1, 0]) and  # x-axis
        (box1[0, 1] >= box2[0, 1] and box1[1, 1] <= box2[1, 1]) and  # y-axis
        (box1[0, 2] >= box2[0, 2] and box1[1, 2] <= box2[1, 2])      # z-axis
    )


box_expanded_min = box[:, 0, :].unsqueeze(1)  # 形状 (N, 1, 3)
box_expanded_max = box[:, 1, :].unsqueeze(1)  # 形状 (N, 1, 3)
qbox_expanded_min = qbox[:, 0, :].unsqueeze(0)  # 形状 (1, N, 3)
qbox_expanded_max = qbox[:, 1, :].unsqueeze(0)  # 形状 (1, N, 3)

# 判断每对 box 和 qbox 是否重叠
overlap_x = (box_expanded_min[:, :, 0] <= qbox_expanded_max[:, :, 0]) & (box_expanded_max[:, :, 0] >= qbox_expanded_min[:, :, 0])
overlap_y = (box_expanded_min[:, :, 1] <= qbox_expanded_max[:, :, 1]) & (box_expanded_max[:, :, 1] >= qbox_expanded_min[:, :, 1])
overlap_z = (box_expanded_min[:, :, 2] <= qbox_expanded_max[:, :, 2]) & (box_expanded_max[:, :, 2] >= qbox_expanded_min[:, :, 2])

# 综合 x, y, z 三个方向的重叠情况
overlap = overlap_x & overlap_y & overlap_z  # 形状 (N, N)，布尔张量

# 统计每个 qbox 找到的重叠数
expected_num_found = overlap.sum(dim=1)


print((expected_num_found - wp.to_torch(num_found)).sum())
