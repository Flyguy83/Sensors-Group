#!/usr/bin/env python3
import time
import roslibpy
import numpy as np

current_pos = None

def joint_state_cb(message):
    global current_pos
    # The JointState message contains 'position' array
    current_pos = list(message['position'])

def call_service(client, service_name):
    """Call a std_srvs/Trigger service and print the response."""
    service = roslibpy.Service(client, service_name, 'std_srvs/Trigger')
    request = roslibpy.ServiceRequest({})
    print(f"[ROS] Calling service: {service_name}")
    result = service.call(request)
    print(f"[ROS] Response: success={result['success']}, message='{result['message']}'")


def move_ur_joint_positions(client,joint_positions, duration=5.0):
    global current_pos
    # client = roslibpy.Ros(host='192.168.27.1', port=9090)  # Replace with your ROS bridge IP

    try:
        # client.run()

        # Subscribe to joint states to get the current position
        listener = roslibpy.Topic(client, '/ur/joint_states', 'sensor_msgs/JointState')
        listener.subscribe(joint_state_cb)

        # Wait until we receive a joint state
        print("[ROS] Waiting for current joint state...")
        start_time = time.time()
        while current_pos is None and time.time() - start_time < 5.0:
            time.sleep(0.05)
        if current_pos is None:
            raise RuntimeError("No joint state received from /ur/joint_states")

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
            '/ur/scaled_pos_joint_traj_controller/command',
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
    client = roslibpy.Ros(host='192.168.27.1', port=9090)  # Replace with your ROS bridge IP
    client.run()
    try:
        # First move
        position_1 = [0.6302809715270996, -1.3169259589961548, 1.2022345701800745, -1.4742978376201172, -1.5368793646441858, 0.0023755908478051424]
        move_ur_joint_positions(client, position_1, duration=5.0)

        # Open gripper after first move
        call_service(client, '/onrobot/close')
        time.sleep(2.0)  # Optional pause between moves

        # Second move
        position_2 = [1.21242094039917, -1.4333710980466385, 0.7061713377581995, -1.7556292019286097, -1.527926270161764, 0.009298055432736874]
        move_ur_joint_positions(client,position_2, duration=5.0)

        # Second move
        position_3 = [1.7473816871643066, -0.9613179129413147, 0.8758967558490198, -1.5557692807963868, -1.5483344236956995, 0.009369985200464725]
        move_ur_joint_positions(client,position_3, duration=5.0)

        # Close gripper after second move
        call_service(client, '/onrobot/open')
        time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n[APP] Interrupted by user.")
