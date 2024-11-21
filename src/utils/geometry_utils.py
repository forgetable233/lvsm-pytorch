import os

import torch
from torch import Tensor

from jaxtyping import Bool, Float, Int64
from einops import einsum, rearrange, reduce, repeat
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)

def check_rot(rot: np.ndarray, right_handed=True, eps=1e-6):
    assert np.allclose(rot.transpose() @ rot, np.eye(3), atol=eps)
    assert np.linalg.det(rot) - 1 < eps * 2
    
    if right_handed:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) - 1.0) < 1e-3
    else:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) + 1.0) < 1e-3


def plot_cooridinate_c2w(poses: np.ndarray, right_handed=True, plot_kf=False, title=None, text="", plot_every_nth_pose=1, ax=None):
    if poses.ndim == 2:
        poses = poses[None, :, :]
    assert poses.ndim == 3
    
    for i in range(poses.shape[0]):
        rot = poses[i, :3, :3]
        check_rot(rot, right_handed=right_handed)
    
        ####### Plotting #######
    new_plot = False
    if ax is None:
        new_plot = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if plot_kf:
        color = ["r", "g", "b"]
        label = ["x kf", "y kf", "z kf"]
    else:
        color = ["c", "m", "y"]
        label = ["x all", "y all", "z all"]
    num_poses = poses.shape[0]

    # Positions
    pos = np.zeros((num_poses, 3))
    pos[:, 0] = np.asarray(poses[:, 0, 3])
    pos[:, 1] = np.asarray(poses[:, 1, 3])
    pos[:, 2] = np.asarray(poses[:, 2, 3])

    # Orientations
    # orientation is given as transposed rotation matrices
    x_ori = []
    y_ori = []
    z_ori = []
    for k in range(num_poses):
        x_ori.append(poses[k, :, 0])
        y_ori.append(poses[k, :, 1])
        z_ori.append(poses[k, :, 2])

    x_mod = 1
    y_mod = 1
    z_mod = 1
    # multiply by 10 to visualize orientation more clearly
    pos = pos * [x_mod, y_mod, z_mod]
    dir_vec_x = pos[0] + x_mod * x_ori[0]
    dir_vec_y = pos[0] + y_mod * y_ori[0]
    dir_vec_z = pos[0] + z_mod * z_ori[0]

    if title is not None:
        ax.plot(
            [pos[0, 0], dir_vec_x[0]],
            [pos[0, 1], dir_vec_x[1]],
            [pos[0, 2], dir_vec_x[2]],
            color=color[0],
            label=label[0],
        )
        ax.plot(
            [pos[0, 0], dir_vec_y[0]],
            [pos[0, 1], dir_vec_y[1]],
            [pos[0, 2], dir_vec_y[2]],
            color=color[1],
            label=label[1],
        )
        ax.plot(
            [pos[0, 0], dir_vec_z[0]],
            [pos[0, 1], dir_vec_z[1]],
            [pos[0, 2], dir_vec_z[2]],
            color=color[2],
            label=label[2],
        )
    else:
        ax.plot([pos[0, 0], dir_vec_x[0]], [pos[0, 1], dir_vec_x[1]], [pos[0, 2], dir_vec_x[2]], color=color[0])
        ax.plot([pos[0, 0], dir_vec_y[0]], [pos[0, 1], dir_vec_y[1]], [pos[0, 2], dir_vec_y[2]], color=color[1])
        ax.plot([pos[0, 0], dir_vec_z[0]], [pos[0, 1], dir_vec_z[1]], [pos[0, 2], dir_vec_z[2]], color=color[2])
    ax.text(pos[0, 0], pos[0, 1], pos[0, 2], text)

    label_every_nth_pose = 2
    # if num_poses / plot_every_nth_pose > 500:
    #   plot_every_nth_pose = int(num_poses / 500)
    #   print(plot_every_nth_pose)
    for k in range(1, num_poses):
        if k % plot_every_nth_pose != 0:
            continue
        dir_vec_x = pos[k] + x_mod * x_ori[k]
        dir_vec_y = pos[k] + y_mod * y_ori[k]
        dir_vec_z = pos[k] + z_mod * z_ori[k]
        ax.plot([pos[k, 0], dir_vec_x[0]], [pos[k, 1], dir_vec_x[1]], [pos[k, 2], dir_vec_x[2]], color=color[0])
        ax.plot([pos[k, 0], dir_vec_y[0]], [pos[k, 1], dir_vec_y[1]], [pos[k, 2], dir_vec_y[2]], color=color[1])
        ax.plot([pos[k, 0], dir_vec_z[0]], [pos[k, 1], dir_vec_z[1]], [pos[k, 2], dir_vec_z[2]], color=color[2])
        if k % label_every_nth_pose == 0:
            ax.text(pos[k, 0], pos[k, 1], pos[k, 2], str(k))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    min = np.min([np.min(pos[:, 1]), np.min(pos[:, 0]), np.min(pos[:, 2])])
    max = np.max([np.max(pos[:, 1]), np.max(pos[:, 0]), np.max(pos[:, 2])])
    ax.set_xlim([min, max])
    ax.set_ylim([min, max])
    ax.set_zlim([min, max])
    # plt.gca().invert_yaxis()

    if title is not None:
        if new_plot:
            plt.legend()
            plt.title(title)
            plt.show()
        else:
            plt.legend()
            plt.title(title)
    else:
        plt.legend()
    return ax     