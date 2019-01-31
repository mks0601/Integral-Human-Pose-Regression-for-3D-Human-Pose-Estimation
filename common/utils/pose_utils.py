import torch
import numpy as np
from config import cfg
import copy

def cam2pixel(cam_coord, f, c):

    x = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
    y = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
    z = cam_coord[..., 2]
    
    return x,y,z

def pixel2cam(pixel_coord, f, c):

    x = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
    y = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
    z = pixel_coord[..., 2]
    
    return x,y,z

def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t

def rigid_align(A, B):
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2

def process_world_coordinate(joint_world, root_idx, joint_num, R, T, f, c):

    # project world coordinates to image space
    joint_cam = np.zeros((joint_num, 3))
    for i in range(joint_num):
        joint_cam[i] = np.dot(R, joint_world[i] - T)
    center_cam = joint_cam[root_idx]

    # Subtract center depth
    joint_img = np.zeros((joint_num, 3))
    joint_img[:, 0], joint_img[:, 1], joint_img[:, 2] = cam2pixel(joint_cam, f, c)
    joint_img[:, 2] = joint_img[:, 2] - center_cam[2]
    joint_vis = np.ones((joint_num,1))
    
    ## bbox3d
    # build 3D bounding box centered on center_cam, sized with bbox_3d_shape
    bbox3d_lt = center_cam - np.array([cfg.bbox_3d_shape[2] / 2, cfg.bbox_3d_shape[1] / 2, 0])
    bbox3d_rb = center_cam + np.array([cfg.bbox_3d_shape[2] / 2, cfg.bbox_3d_shape[1] / 2, 0])

    # back-project 3D BBox to 2D image
    bbox2d_l, bbox2d_t, _ = cam2pixel(bbox3d_lt, f, c)
    bbox2d_r, bbox2d_b, _ = cam2pixel(bbox3d_rb, f, c)
    bbox = np.array([bbox2d_l, bbox2d_t, bbox2d_r-bbox2d_l+1, bbox2d_b-bbox2d_t+1])

    return joint_img, joint_cam, joint_vis, center_cam, bbox

def warp_coord_to_original(joint_out, bbox, center_cam):

    # joint_out: output from soft-argmax
    x = joint_out[:, 0] / cfg.output_shape[1] * bbox[2] + bbox[0]
    y = joint_out[:, 1] / cfg.output_shape[0] * bbox[3] + bbox[1]
    z = (joint_out[:, 2] / cfg.depth_dim * 2. - 1.) * (cfg.bbox_3d_shape[0]/2.) + center_cam[2]

    return x, y, z

def fliplr_joints(_joints, width, matched_parts):
    """
    flip coords
    joints: numpy array, nJoints * dim, dim == 2 [x, y] or dim == 3  [x, y, z]
    width: image width
    matched_parts: list of pairs
    """
    joints = _joints.copy()
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()

    return joints

def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped

