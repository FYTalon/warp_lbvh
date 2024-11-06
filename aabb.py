from warp_lbvh.common import *

aabb = wp.mat(shape=(2, 3), dtype=wpfloat)

@wp.func
def get_row(box: aabb, row: int):
    return vec3(box[row, 0], box[row, 1], box[row, 2])

@wp.func
def intersects(lhs: aabb, rhs: aabb) -> bool:
    # lhs is the aabb matrix for the first box (2x3 matrix)
    # rhs is the aabb matrix for the second box (2x3 matrix)

    lhs_upper = get_row(lhs, 1)
    lhs_lower = get_row(lhs, 0)

    rhs_upper = get_row(rhs, 1)
    rhs_lower = get_row(rhs, 0)

    # Check for intersection based on axis-aligned bounding box rules
    if lhs_upper.x < rhs_lower.x or rhs_upper.x < lhs_lower.x:
        return False
    if lhs_upper.y < rhs_lower.y or rhs_upper.y < lhs_lower.y:
        return False
    if lhs_upper.z < rhs_lower.z or rhs_upper.z < lhs_lower.z:
        return False

    return True

@wp.func
def merge(lhs: aabb, rhs: aabb) -> aabb:
    # Extract the upper and lower bounds from lhs and rhs
    lhs_upper = get_row(lhs, 1)  # vec3 for lhs upper bound
    lhs_lower = get_row(lhs, 0)  # vec3 for lhs lower bound
    rhs_upper = get_row(rhs, 1)  # vec3 for rhs upper bound
    rhs_lower = get_row(rhs, 0)  # vec3 for rhs lower bound

    # Create a new aabb for the merged result
    merged_upper = vec3(
        wp.max(lhs_upper.x, rhs_upper.x),
        wp.max(lhs_upper.y, rhs_upper.y),
        wp.max(lhs_upper.z, rhs_upper.z)
    )

    merged_lower = vec3(
        wp.min(lhs_lower.x, rhs_lower.x),
        wp.min(lhs_lower.y, rhs_lower.y),
        wp.min(lhs_lower.z, rhs_lower.z)
    )

    # Return the merged AABB as a 2x3 matrix
    ret = aabb(wpfloat(0.0))
    ret[0, 0] = merged_lower.x
    ret[0, 1] = merged_lower.y
    ret[0, 2] = merged_lower.z
    ret[1, 0] = merged_upper.x
    ret[1, 1] = merged_upper.y
    ret[1, 2] = merged_upper.z

    return ret

@wp.func
def mindist(lhs: aabb, rhs: vec3) -> wpfloat:
    # Extract the upper and lower bounds from lhs
    lhs_upper = get_row(lhs, 1)
    lhs_lower = get_row(lhs, 0)

    # Calculate the minimum distance between point and AABB
    dx = wp.min(lhs_upper.x, wp.max(lhs_lower.x, rhs.x)) - rhs.x
    dy = wp.min(lhs_upper.y, wp.max(lhs_lower.y, rhs.y)) - rhs.y
    dz = wp.min(lhs_upper.z, wp.max(lhs_lower.z, rhs.z)) - rhs.z

    # Return squared distance
    return dx * dx + dy * dy + dz * dz

@wp.func
def minmaxdist(lhs: aabb, rhs: vec3) -> wpfloat:
    # Extract the lower and upper bounds from lhs
    lhs_upper = get_row(lhs, 1)
    lhs_lower = get_row(lhs, 0)

    # Calculate the squared distances for rm_sq and rM_sq
    rm_sq = vec3(
        (lhs_lower.x - rhs.x) * (lhs_lower.x - rhs.x),
        (lhs_lower.y - rhs.y) * (lhs_lower.y - rhs.y),
        (lhs_lower.z - rhs.z) * (lhs_lower.z - rhs.z)
    )

    rM_sq = vec3(
        (lhs_upper.x - rhs.x) * (lhs_upper.x - rhs.x),
        (lhs_upper.y - rhs.y) * (lhs_upper.y - rhs.y),
        (lhs_upper.z - rhs.z) * (lhs_upper.z - rhs.z)
    )

    # Conditional swap based on the mid-point of the AABB
    if (lhs_upper.x + lhs_lower.x) * 0.5 < rhs.x:
        rm_sq.x, rM_sq.x = rM_sq.x, rm_sq.x

    if (lhs_upper.y + lhs_lower.y) * 0.5 < rhs.y:
        rm_sq.y, rM_sq.y = rM_sq.y, rm_sq.y

    if (lhs_upper.z + lhs_lower.z) * 0.5 < rhs.z:
        rm_sq.z, rM_sq.z = rM_sq.z, rm_sq.z

    # Calculate dx, dy, dz based on the modified rm_sq and rM_sq
    dx = rm_sq.x + rM_sq.y + rM_sq.z
    dy = rM_sq.x + rm_sq.y + rM_sq.z
    dz = rM_sq.x + rM_sq.y + rm_sq.z

    # Return the minimum of dx, dy, dz
    return wp.min(dx, wp.min(dy, dz))

@wp.func
def centroid(box: aabb) -> vec3:
    # Extract the upper and lower bounds from the box
    # upper = box.get_row(0)
    upper = get_row(box, 1)
    lower = get_row(box, 0)

    # Calculate the centroid
    c = vec3(
        (upper.x + lower.x) * 0.5,
        (upper.y + lower.y) * 0.5,
        (upper.z + lower.z) * 0.5
    )

    return c
