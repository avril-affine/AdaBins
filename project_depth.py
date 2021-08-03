import numpy as np
from PIL import Image


def col(a: np.ndarray) -> np.ndarray:
    return a.reshape(1, -1)

def row(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1, 1)

def project_depth(depth: Image, max_depth: int = 10) -> Image:
    depth = np.asarray(depth, np.float32)
    import ipdb; ipdb.set_trace()
    H, W = depth.shape

    xx, yy = np.meshgrid(range(W), range(H))

    # Color camera parameters
    fx_rgb = 5.1930334103339817e02
    fy_rgb = 5.1816401430246583e02
    cx_rgb = 3.2850951551345941e02
    cy_rgb = 2.5282555217253503e02

    # Depth camera parameters
    fx_d = 5.7616540758591043e02
    fy_d = 5.7375619782082447e02
    cx_d = 3.2442516903961865e02
    cy_d = 2.3584766381177013e02

    # Rotation matrix
    R = np.linalg.inv(
        [
            [9.9998579449446667e-01, 3.4203777687649762e-03, -4.0880099301915437e-03],
            [-3.4291385577729263e-03, 9.9999183503355726e-01, -2.1379604698021303e-03],
            [4.0806639192662465e-03, 2.1519484514690057e-03, 9.9998935859330040e-01],
        ]
    )

    # Translation vector.
    T = -np.array([[2.2142187053089738e-02], [-1.4391632009665779e-04], [-7.9356552371601212e-03]])

    fc_d = np.array([fx_d, fy_d])
    cc_d = np.array([cx_d, cy_d])

    fc_rgb = np.array([fx_rgb, fy_rgb])
    cc_rgb = np.array([cx_rgb, cy_rgb])

    # 1. raw depth --> absolute depth in meters.
    depth2 = 0.3513e3 / (1.0925e3 - depth)  # nathan's data

    depth2[depth2 > max_depth] = max_depth
    depth2[depth2 < 0] = 0

    # 2. points in depth image to 3D world points:
    x3 = (xx - cc_d[0]) * depth2 / fc_d[0]
    y3 = (yy - cc_d[1]) * depth2 / fc_d[1]
    z3 = depth2

    # 3. now rotate & translate the 3D points
    p3 = np.concatenate([col(x3), col(y3), col(z3)])
    p3_new = np.matmul(R, p3) + T * np.ones((1, p3.shape[1]))

    x3_new = p3_new[0, :].T.reshape(H, W)
    y3_new = p3_new[1, :].T.reshape(H, W)
    z3_new = p3_new[2, :].T.reshape(H, W)

    # 4. project into rgb coordinate frame
    x_proj = (x3_new * fc_rgb[0] / z3_new) + cc_rgb[0]
    y_proj = (y3_new * fc_rgb[1] / z3_new) + cc_rgb[1]

    # now project back to actual image

    x_proj = np.round(x_proj)
    y_proj = np.round(y_proj)
    x_proj_flat, y_proj_flat = x_proj.flatten(), y_proj.flatten()

    g_ind, = np.where(
        (x_proj_flat > 0) & (x_proj_flat < depth.shape[1]) & (y_proj_flat > 0) & (y_proj_flat < depth.shape[0])
    )

    depthProj = np.zeros_like(depth)
    depth2_flat = depth2.flatten()
    order = np.argsort(-depth2_flat[g_ind])
    depth_sorted = np.take(depth2_flat, order)

    # z-buffer projection
    indices = g_ind[order]
    y_ind, x_ind = np.take(y_proj_flat, indices), np.take(x_proj_flat, indices)
    flat_indices = (y_ind * W + x_ind).astype(np.int16)
    np.put(depthProj, flat_indices, depth_sorted)

    # Fix weird values...
    q = depthProj > max_depth
    depthProj[q] = max_depth

    q = depthProj < 0
    depthProj[q] = 0

    q = np.isnan(depthProj)
    depthProj[q] = 0

    # depthProj = np.expand_dims(depthProj, axis=-1)
    import ipdb; ipdb.set_trace()
    return Image.fromarray(depthProj)
