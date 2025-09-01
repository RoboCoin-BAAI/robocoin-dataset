import numpy as np
from scipy.spatial.transform import Rotation as r


def cm2m(input: np.ndarray) -> np.ndarray:
    return input / 100


def mm2m(input: np.ndarray) -> np.ndarray:
    return input / 1000


def quat_xyzw_2_euler_xyz(input: np.ndarray) -> np.ndarray:
    return r.from_quat(input).as_euler("xyz")


def quat_wxyz_2_euler_xyz(input: np.ndarray) -> np.ndarray:
    quat_xyzw = np.roll(input, -1)
    return r.from_quat(quat_xyzw).as_euler("xyz")


def rot6d_to_matrix(vec6: np.ndarray) -> np.ndarray:
    """
    将 6D rotation 转为 3x3 旋转矩阵
    vec6: (6,)
    返回: (3, 3)
    """
    # 单个样本
    v1 = vec6[:3]  # x-axis
    v2 = vec6[3:]  # y-axis

    # 归一化 x-axis
    x_axis = v1 / np.linalg.norm(v1)

    # 投影并归一化 y-axis
    v2_proj = v2 - np.dot(v2, x_axis) * x_axis
    y_axis = v2_proj / np.linalg.norm(v2_proj)

    # z-axis = x × y
    z_axis = np.cross(x_axis, y_axis)

    return np.column_stack([x_axis, y_axis, z_axis])


def rot6d_to_euler_xyz(vec6: np.ndarray) -> np.ndarray:
    """
    将 6D rotation 转为欧拉角
    vec6: (6,) 或 (N, 6)
    """
    return r.from_matrix(rot6d_to_matrix(vec6)).as_euler("xyz")


def degree2rad(input: np.ndarray) -> np.ndarray:
    return np.deg2rad(input)


spatial_covertor_funcs = {
    "mm2m": mm2m,
    "cm2m": cm2m,
    "quat_wxyz_2_rot_xyz": quat_xyzw_2_euler_xyz,
    "rot6d_to_euler_xyz": rot6d_to_euler_xyz,
    "degree2rad": degree2rad,
}
