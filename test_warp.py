import torch
import warp as wp

@wp.kernel
def func():
    a = wp.mat33(0.0)

    print(a.get_row(0))

wp.init()

wp.launch(
    kernel=func,
    dim=(3,),
    inputs=[]
)
