import numpy as np
import cv2 as cv2
import pyrealsense2 as rs
from pathlib import Path

def FuncLx(x, y, Z):
    Lx = np.zeros((2, 6), dtype=float)
    Lx[0, 0] = -1.0 / Z
    Lx[0, 1] = 0.0
    Lx[0, 2] = x / Z
    Lx[0, 3] = x * y
    Lx[0, 4] = -(1.0 + x**2)
    Lx[0, 5] = y

    Lx[1, 0] = 0.0
    Lx[1, 1] = -1.0 / Z
    Lx[1, 2] = y / Z
    Lx[1, 3] = 1.0 + y**2
    Lx[1, 4] = -x * y
    Lx[1, 5] = -x
    return Lx

Z = 50
Lambda = 0.5

Target = np.array([
    [0.0,   0.0],
    [800.0, 0.0],
    [0.0, 800.0]
], dtype=float)

PATTERN_SIZE = (9, 6)  # (cols, rows)

RES = (640, 480)
FPS = 30

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, RES[0], RES[1], rs.format.bgr8, FPS)
profile = pipeline.start(config)

color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

target_norm = np.empty((3,2), float)
target_norm[:,0] = (Target[:,0] - cx) / fx
target_norm[:,1] = (Target[:,1] - cy) / fy

Lx = np.vstack([FuncLx(target_norm[i,0], target_norm[i,1], Z) for i in range(3)]) #CHANGE FOR TARGET FEATURE MATRIX SIZE

while(1):
    try:
        # Grab one frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError('No colour frames received')

        color = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ret or corners is None or len(corners) != PATTERN_SIZE[0] * PATTERN_SIZE[1]:
            raise RuntimeError('Checkerboard not found or wrong pattern size.')

        cv2.cornerSubPix(
            gray, corners, (5,5), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        )  # corners: (N,1,2)

        # Normalize pixel coordinates
        uv = corners.reshape(-1, 2)
        x = (uv[:, 0] - cx) / fx
        y = (uv[:, 1] - cy) / fy
        obs_norm = np.column_stack((x, y))
    finally:
        print('Coordinates received')    

    e2 = obs_norm - target_norm 
    e = e2.T.reshape(-1, 1, order='F')  

    Lx_pinv = np.linalg.pinv(Lx)  
    Vc = -Lambda * (Lx_pinv @ e) 
