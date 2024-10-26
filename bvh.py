from bvh_utils import *

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
        self.objects = objects.contiguous().cuda()
        self.AABBs = torch.empty(0)
        self.nodes = torch.empty(0)


    def contruct(self):

        if self.objects.shape[0] == 0:
            return

        num_objects = self.objects.shape[0]
        num_internal_nodes = num_objects - 1
        num_nodes = num_objects * 2 - 1

        self.AABBs = wp.zeros(shape=num_nodes, dtype=aabb)
        AABBs_t = wp.to_torch(self.AABBs)

        default_aabb = aabb(0.0)
        default_aabb.set_row(0, vec3(-infty))
        default_aabb.set_row(1, vec3(infty))

        AABBs_t[num_internal_nodes:] = self.objects

        aabb_whole_t = torch.stack(
            (
                torch.max(AABBs_t[:, 0, :], dim=0).values,
                torch.min(AABBs_t[:, 1, :], dim=0).values
            )
        )

        aabb_whole = aabb(
            aabb_whole_t[0, 0], aabb_whole_t[0, 1], aabb_whole_t[0, 2],
            aabb_whole_t[1, 0], aabb_whole_t[1, 1], aabb_whole_t[1, 2],
        )

        mortons = wp.zeros(shape=num_objects, dtype=wp.int32)

        tmp = wp.from_torch(self.objects, dtype=aabb)

        wp.launch(
            kernel=transform_aabb_morton_kernel,
            dim=(num_objects,),
            inputs=[
                tmp,
                mortons,
                aabb_whole,
            ]
        )
        wp.synchronize()

        mortons_t = wp.to_torch(mortons)

        mortons_sort_t, indices_t = torch.sort(mortons_t)
        indices_t = indices_t.to(dtype=torch.int32)

        AABBs_t[num_internal_nodes:] = AABBs_t[num_internal_nodes:][indices_t]

        mortons64 = wp.zeros(shape=num_objects, dtype=wpuint64)

        indices = wp.from_torch(indices_t, dtype=wpuint32)

        wp.launch(
            kernel=transform_morton64_kernel,
            dim=(num_objects,),
            inputs=[
                wp.from_torch(mortons_sort_t),
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
        wp.synchronize()


