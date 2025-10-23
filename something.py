#!/usr/bin/env python3
import time
import numpy as np
import cv2
import pyrealsense2 as rs
import roslibpy
import roboticstoolbox as rtb
from roboticstoolbox import models
from spatialmath import SE3
from rtde_control import RTDEControlInterface

# ------------------------- CAMERA + VISUAL SERVOING SETUP ------------------------- #

def FuncLx(x, y, Z): #Computing the Image Jacobian for one image point
    #Vc = [vx, vy, vz, wx, wy, wz]^T
    # Image Feature Velocity [x dot, y dot] = [vx, vy, vz, wx, wy, wz]^T  * Lx
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

Z = 0.4 #Assumed constant depth of the target points in m
Lambda = 0.5 # Visual servoing gain

r = models.DH.UR3e()

Target = np.array([
    [444.16614,  127.190315],
    [451.59695,  358.1429],
    [250.18605,  357.43393],
    [210.23355,  357.4991]
    ], dtype=float) # Four corners for simplicity?

PATTERN_SIZE = (7, 7) # (cols, rows)
PATTERN_INDEX = (0,6,41,48) # Indexes of top corners based on pattern size

RESOLUTION = (640, 480)
FPS = 30

# ----------Initialising the Intel RealSense coulor stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS)
profile = pipeline.start(config)
#----------------------------------------------------------

color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_stream.get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

target_norm = np.empty((4, 2), float)
target_norm[:, 0] = (Target[:, 0] - cx) / fx
target_norm[:, 1] = (Target[:, 1] - cy) / fy #Normalised target feature coordinates

Lx = np.vstack([FuncLx(target_norm[i, 0], target_norm[i, 1], Z) for i in range(4)])

# ------------------------- ROS + ROBOT CONTROL SETUP ------------------------- #

current_pos = None

def joint_state_cb(message):
    global current_pos
    current_pos = list(message['position'])

# ------------------------- MAIN LOOP ------------------------- #

def move_ur_joint_positions(client,joint_positions, duration=5.0):
    global current_pos
    # client = roslibpy.Ros(host='192.168.27.1', port=9090)  # Replace with your ROS bridge IP

    try:
        # client.run()

        # Subscribe to joint states to get the current position
        listener = roslibpy.Topic(client, '/joint_states', 'sensor_msgs/JointState')
        listener.subscribe(joint_state_cb)

        # Wait until we receive a joint state
        print("[ROS] Waiting for current joint state...")
        start_time = time.time()
        while current_pos is None and time.time() - start_time < 5.0:
            time.sleep(0.05)
        if current_pos is None:
            raise RuntimeError("No joint state received from /joint_states")

        print(f"[ROS] Current joint positions: {current_pos}")

        # Build a JointTrajectory message for the scaled_pos_joint_traj_controller
        joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        trajectory_msg = {
            'joint_names': joint_names,
            'points': [
                {
                    'positions': current_pos,
                    'time_from_start': {'secs': 0, 'nsecs': 0}
                },
                {
                    'positions': joint_positions, #First Position
                    'time_from_start': {
                        'secs': int(duration),
                        'nsecs': int((duration - int(duration)) * 1e9)
                    }
                }
            ]
        }

        # Publish to the controller's /command topic
        topic = roslibpy.Topic(
            client,
            '/scaled_pos_joint_traj_controller/command',
            'trajectory_msgs/JointTrajectory'
        )
        topic.advertise()
        topic.publish(roslibpy.Message(trajectory_msg))
        print("[ROS] Trajectory published.")

        # Wait for motion to complete
        time.sleep(duration + 1.0)

        topic.unadvertise()


        
        listener.unsubscribe()

    finally:
        pass
        # client.terminate()
        # print("[ROS] Disconnected from rosbridge.")


if __name__ == '__main__':
    client = roslibpy.Ros(host='192.168.27.1', port=9090)
    client.run()

    #Robot's state subscriber
    listener = roslibpy.Topic(client, '/ur/joint_states', 'sensor_msgs/JointState')
    listener.subscribe(joint_state_cb)

    print("[APP] Starting visual servoing control loop... Press Ctrl+C to stop.")
    time.sleep(2.0)

    try:
        while True: #Finds/Detects the checkerboard
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError('No colour frames received')
                continue

            color = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, PATTERN_SIZE,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if not ret or corners is None:
                print("[CAMERA] Checkerboard not found.")
                continue


            # Refine corner locations
            cv2.cornerSubPix(gray, corners, (5,5), (-1,-1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))

            edge_corners = np.array([corners[idx, 0, :] for idx in PATTERN_INDEX])
            
            # Normalise pixel coordinates in image frame with camera intrinsics
            x = (edge_corners[:, 0] - cx) / fx
            y = (edge_corners[:, 1] - cy) / fy
            obs_norm = np.column_stack((x, y))

            # Visual servoing control law
            e2 = obs_norm - target_norm 
            e = e2.T.reshape(-1, 1, order='F')  
        
            Lx_pinv = np.linalg.pinv(Lx)
            Vc = -Lambda * (Lx_pinv @ e) 
            # Vc = -lambda * Lx+ * e 
            #Vc is 6x1 velocity command for the camera frame
            #The - ensures the robot moves to reduce the error

            r.q = current_pos
            J2 = r.jacobe(current_pos)
            Jinv = np.linalg.pinv(J2)
            qp = Jinv @ Vc

            goal_pos = current_pos+qp
            
            print(f"[CTRL] Velocity command: {Vc.T}")

    except KeyboardInterrupt:
        print("\n[APP] Stopped by user.")
    finally:
        listener.unsubscribe()
        pipeline.stop()
        client.terminate()
        print("[APP] Shutdown complete.")
