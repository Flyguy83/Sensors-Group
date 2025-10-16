# -----------------------------------------------------------------------------------#
# LabAssignment1: UR3 Brick Stacking Simulation (with qlim enforcement)
# -----------------------------------------------------------------------------------#

import numpy as np
import swift
import os

from spatialmath.base import *
from spatialmath      import SE3
from spatialgeometry  import Cuboid, Mesh
from roboticstoolbox  import jtraj
from scipy.spatial    import ConvexHull
from ir_support       import UR3
from math             import pi


# -----------------------------------------------------------------------------------#
class LabAssignment1():
    # Set Up: Environment Set Up + Parameter Definitions + Brick I/F Position Definitions
    def __init__(self):
        # Creating and launching swift environment
        self.env = swift.Swift()
        self.env.launch(realtime=True)

        # Brick parameters
        y_pos            = -0.4
        spacing_final    = 0.15
        self.brick_h     = 0.032
        self.brick_l     = 0.1
        self.brick_w     = 0.05
        num_stacks       = 3
        bricks_per_stack = 3
        brick_start      = 0

        # Gripper parameters
        self.open_pos          = 0.063 * 2
        self.closed_pos        = 0.040 * 2
        self.gripperbaseheight = 0.08
        self.gripperlength     = self.brick_h
        fingerscale            = [0.02, 0.02, self.gripperlength]

        # Fire Extinguisher and Emergency Button 
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Generating the bricks
        dae_path = os.path.join(current_dir, "HalfSizedRedGreenBrick.dae")
        self.bricks = []
        self.brick_initial_pos = []
        
        # Create the environment objects
        self.gripper_base = Cuboid(scale=[0.15, 0.065, self.gripperbaseheight], color=(0.3, 0.3, 0.3))
        self.finger1      = Cuboid(scale=fingerscale, color=(0.7, 0.7, 0.7))
        self.finger2      = Cuboid(scale=fingerscale, color=(0.7, 0.7, 0.7))
        self.robot        = UR3() 

        # ---------------------- Joint Limits ---------------------- #
        # Define full qlim for each joint (UR3 typical ranges in radians)
        self.robot.links[1].qlim = np.array([-pi/2, pi/2])   # Shoulder
        self.robot.links[2].qlim = np.array([-pi/2, pi/2])   # Elbow
        # ----------------------------------------------------------- #

        # Add objects to the environment
        self.robot.add_to_env(self.env)
        self.update_gripper(self.open_pos)
        self.env.add(self.gripper_base)
        self.env.add(self.finger1)
        self.env.add(self.finger2)

        # Initial Brick Positions (hardcoded) for Viva Demo
        hardcodedpos = [
            [ 0.00, -0.4, 0], # Brick 0
            [-0.10, -0.4, 0], # Brick 1
            [ 0.10, -0.4, 0]  # Brick 2
        ]

        for pos in hardcodedpos:
            brick_T = SE3(pos)
            brick = Mesh(filename=dae_path, pose=brick_T)
            self.bricks.append(brick)
            self.brick_initial_pos.append(pos)
            self.env.add(brick)

        # Final Brick Positions
        self.wall_pos = [
            [ 0.00, 0.4,              0], # Brick 0
            [ 0.00, 0.4,   self.brick_h], # Brick 1
            [ 0.00, 0.4, 2*self.brick_h]  # Brick 2
        ]


    # ---------------------------------------------------------------- #
    # Helper: Clamp any joint vector to within robot's defined qlims
    # ---------------------------------------------------------------- #
    def clamp_q(self, q):
        qmin = np.array([link.qlim[0] for link in self.robot.links])
        qmax = np.array([link.qlim[1] for link in self.robot.links])
        return np.clip(q, qmin, qmax)

    # ---------------------------------------------------------------- #
    # Defining the gripper position and updating positions of gripper parts
    # ---------------------------------------------------------------- #
    def update_gripper(self, sep):
        ee_tr = self.robot.fkine(self.robot.q)
        self.gripper_base.T = ee_tr * SE3(0, 0, -self.gripperbaseheight / 2)
        self.finger1.T      = ee_tr * SE3(sep / 2, 0, self.gripperlength / 2)
        self.finger2.T      = ee_tr * SE3(-sep / 2, 0, self.gripperlength / 2)

    # ---------------------------------------------------------------- #
    # Running the simulation to pick up and place the bricks correctly
    # ---------------------------------------------------------------- #
    def run_simulation(self):
        self.picked_brick = None
        gripperorientation = SE3.OA([0, 1, 0], [0, 0, -1])
        steps = 50
        brickgripperoffset = SE3(0, 0, 0)
        Successfullyplacedbrick = 0

        for brick_idx in range(3):
            print(f"\n--- Processing brick {brick_idx} ---")

            # --- Move above brick ---
            pick_pos_above = self.brick_initial_pos[brick_idx].copy()
            pick_pos_above[2] += 0.1
            T_pick_above = SE3(pick_pos_above) * gripperorientation
            print(f"Transform to move above brick {brick_idx}:\n{T_pick_above}")
            sol_above = self.robot.ikine_LM(T_pick_above, q0=self.robot.q, joint_limits=True)

            if not sol_above.success:
                print(f"Skipping brick {brick_idx}: cannot reach pick_pos_above.")
                continue

            traj = jtraj(self.robot.q, sol_above.q, steps)
            for qk in traj.q:
                qk = self.clamp_q(qk)
                self.robot.q = qk
                self.update_gripper(self.open_pos)
                if self.picked_brick:
                    self.picked_brick.T = self.robot.fkine(self.robot.q) * brickgripperoffset
                self.env.step(0.02)

            # --- Move to grasp brick ---
            grasp_pos = self.brick_initial_pos[brick_idx].copy()
            T_grasp = SE3(grasp_pos) * SE3(0, 0, 0.04) * gripperorientation
            print(f"Transform to grasp brick {brick_idx}:\n{T_grasp}")
            sol_grasp = self.robot.ikine_LM(T_grasp, q0=self.robot.q, joint_limits=True)

            if not sol_grasp.success:
                print(f"Skipping brick {brick_idx}: cannot reach grasp_pos.")
                continue

            q_traj = jtraj(self.robot.q, sol_grasp.q, steps).q
            for k in range(steps):
                qk = self.clamp_q(q_traj[k])
                self.robot.q = qk
                frac = k / (steps - 1)
                sep = self.open_pos - frac * (self.open_pos - self.closed_pos)
                self.update_gripper(sep)
                self.env.step(0.02)

            # --- Pick up the brick ---
            self.picked_brick = self.bricks[brick_idx]
            traj = jtraj(self.robot.q, sol_above.q, steps)
            for qk in traj.q:
                qk = self.clamp_q(qk)
                self.robot.q = qk
                self.update_gripper(self.closed_pos)
                self.picked_brick.T = self.robot.fkine(self.robot.q) * brickgripperoffset
                self.env.step(0.02)

            # --- Move above placement ---
            place_pos_above = self.wall_pos[Successfullyplacedbrick].copy()
            place_pos_above[2] += 0.1
            T_place_above = SE3(place_pos_above) * gripperorientation
            print(f"Above placement pos transform for brick {brick_idx}:\n{T_place_above}")
            sol_place_above = self.robot.ikine_LM(T_place_above, q0=self.robot.q, joint_limits=True)

            if not sol_place_above.success:
                print(f"Skipping brick {brick_idx}: cannot reach place_pos_above.")
                self.picked_brick = None
                continue

            traj = jtraj(self.robot.q, sol_place_above.q, steps)
            for qk in traj.q:
                qk = self.clamp_q(qk)
                self.robot.q = qk
                self.update_gripper(self.closed_pos)
                self.picked_brick.T = self.robot.fkine(self.robot.q) * brickgripperoffset
                self.env.step(0.02)

            # --- Place brick ---
            place_pos = self.wall_pos[Successfullyplacedbrick].copy()
            T_place = SE3(place_pos) * SE3(0, 0, self.brick_h / 2) * gripperorientation
            print(f"Transform to place brick {brick_idx}:\n{T_place}")
            sol_place = self.robot.ikine_LM(T_place, q0=self.robot.q, joint_limits=True)

            if not sol_place.success:
                print(f"Skipping brick {brick_idx}: cannot reach place_pos.")
                self.picked_brick = None
                continue

            q_traj = jtraj(self.robot.q, sol_place.q, steps).q
            for k in range(steps):
                qk = self.clamp_q(q_traj[k])
                self.robot.q = qk
                frac = k / (steps - 1)
                sep = self.closed_pos + frac * (self.open_pos - self.closed_pos)
                self.update_gripper(sep)
                if k < steps - 1:
                    self.picked_brick.T = self.robot.fkine(self.robot.q)
                self.env.step(0.02)

            # --- Return to above placement ---
            traj = jtraj(self.robot.q, sol_place_above.q, steps)
            for qk in traj.q:
                qk = self.clamp_q(qk)
                self.robot.q = qk
                self.update_gripper(self.open_pos)
                self.env.step(0.02)

            # Increment placement counter only after a successful place
            Successfullyplacedbrick += 1
            self.picked_brick = None

        print(f"\nSimulation finished. Successfully placed {Successfullyplacedbrick} bricks.")

# ---------------------------------------------------------------------------------------#
# Main loop
if __name__ == "__main__":
    assignment = LabAssignment1()
    input("Press Enter to start the simulation...")
    assignment.run_simulation()
    print("Simulation complete.")
    assignment.env.hold()
