from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    K: np.ndarray
    dist: np.ndarray


@dataclass
class StereoSettings:
    left: CameraIntrinsics
    right: CameraIntrinsics
    R: np.ndarray
    T: np.ndarray


def load_camera_intrinsics(calib_path, camera_suffix):
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open {calib_path}")
    K = fs.getNode(f"K_{camera_suffix}").mat()
    dist = fs.getNode(f"dist_{camera_suffix}").mat()
    fs.release()
    if K is None or dist is None:
        raise RuntimeError(
            f"K_{camera_suffix}/dist_{camera_suffix} not found in {calib_path}"
        )
    return CameraIntrinsics(K=K, dist=dist)


def load_stereo_calibration(calib_path):
    left = load_camera_intrinsics(calib_path, "G")
    right = load_camera_intrinsics(calib_path, "D")
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    fs.release()
    if R is None or T is None:
        raise RuntimeError(f"R/T not found in {calib_path}")
    return StereoSettings(left=left, right=right, R=R, T=T)


def undistort_point(cx, cy, K, dist):
    pts = np.array([[[cx, cy]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, K, dist, P=K)
    return float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1])


def save_stereo_calibration(output_path, stereo, **extra):
    fs = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
    fs.write("K_G", stereo.left.K)
    fs.write("dist_G", stereo.left.dist)
    fs.write("K_D", stereo.right.K)
    fs.write("dist_D", stereo.right.dist)
    fs.write("R", stereo.R)
    fs.write("T", stereo.T)
    for key, val in extra.items():
        fs.write(key, val)
    fs.release()
