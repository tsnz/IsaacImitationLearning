import numpy as np

# adapted from isaaclab.utils.math

pi = np.pi


def euler_xyz_from_quat(quat: np.ndarray) -> np.ndarray:
    """Convert rotations given as quaternions to Euler angles in radians.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (4).

    Returns:
        An array containing roll-pitch-yaw.

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    q_w, q_x, q_y, q_z = quat[0], quat[1], quat[2], quat[3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = np.arctan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = np.where(np.abs(sin_pitch) >= 1, np.copysign(np.pi / 2.0, sin_pitch), np.arcsin(sin_pitch))

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = np.arctan2(sin_yaw, cos_yaw)

    return np.asarray([roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)])


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (4).
        q2: The second quaternion in (w, x, y, z). Shape is (4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.asarray([w, x, y, z])


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians).
        pitch: Rotation around y-axis (in radians).
        yaw: Rotation around z-axis (in radians).

    Returns:
        The quaternion in (w, x, y, z).
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.asarray([qw, qx, qy, qz])
