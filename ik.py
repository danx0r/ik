import math
import mujoco
import numpy as np
import glfw
import argparse
import time
import sys, select
from scipy.spatial.transform import Rotation as R
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

MAXDIST = 2

f = open("claude3d.xml")
MODEL_XML = f.read()
f.close()

def kbhit():
    r = select.select([sys.stdin], [], [], 0.01)
    return len(r[0]) > 0

class InteractiveScene:
    def __init__(self):
        # First initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        # Store important instance variables
        self.button_left = False
        self.button_right = False
        self.last_x = 0
        self.last_y = 0
        self.rotation_speed = 5.0
        self.pan_speed = 0.1
        self.zoom_speed = 0.5

        # Create window
        self.window = glfw.create_window(1200, 900, "MuJoCo Demo", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")

        glfw.make_context_current(self.window)

        # Set up callbacks
        glfw.set_mouse_button_callback(self.window, self._mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move)
        glfw.set_scroll_callback(self.window, self._scroll)

        # Load model and create data
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)

        # Initialize camera
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 4.5
        self.cam.azimuth = 90.0
        self.cam.elevation = -10.0
        self.cam.lookat[0] = 0
        self.cam.lookat[1] = 0
        self.cam.lookat[2] = 0

        # Create scene and context
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    def _mouse_button(self, window, button, act, mods):
        self.button_left = (glfw.MOUSE_BUTTON_LEFT == button and glfw.PRESS == act)
        self.button_right = (glfw.MOUSE_BUTTON_RIGHT == button and glfw.PRESS == act)
        x, y = glfw.get_cursor_pos(window)
        self.last_x = x
        self.last_y = y

    def _mouse_move(self, window, xpos, ypos):
        dx = xpos - self.last_x
        dy = ypos - self.last_y

        if self.button_left:
            self.cam.azimuth += dx * -0.5
            self.cam.elevation = max(min(self.cam.elevation - dy * 0.5, 90), -90)
        elif self.button_right:
            self.cam.lookat[0] -= dx * 0.01 * self.cam.distance
            self.cam.lookat[1] += dy * 0.01 * self.cam.distance

        self.last_x = xpos
        self.last_y = ypos

    def _scroll(self, window, xoffset, yoffset):
        self.cam.distance = max(self.cam.distance - yoffset * 0.5, 0.5)

    def run(self, steps=999999, j1=None, j2=None, j3=None, j4=None, j5=None, j6=None, x=None, y=None, z=None, qw=None, qx=None, qy=None, qz=None, render=True):
        while steps > 0 and not glfw.window_should_close(self.window):
            if kbhit():
                input()
                break
            steps -= 1
            time_prev = self.data.time

            if x is not None:
                self.data.mocap_pos[0]=[x, y, z]

            if qw is not None:
                self.data.mocap_quat[0]=[qw, qx, qy, qz]

            if j1 is not None:
                self.data.actuator('j1').ctrl[0] = j1
            if j2 is not None:
                self.data.actuator('j2').ctrl[0] = j2
            if j3 is not None:
                self.data.actuator('j3').ctrl[0] = j3
            if j4 is not None:
                self.data.actuator('j4').ctrl[0] = j4
            if j5 is not None:
                self.data.actuator('j5').ctrl[0] = j5
            if j6 is not None:
                self.data.actuator('j6').ctrl[0] = j6

            while (self.data.time - time_prev < 1.0/60.0):
                mujoco.mj_step(self.model, self.data)

            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

            # Update scene and render
            mujoco.mj_forward(self.model, self.data)
            mujoco.mjv_updateScene(
                self.model, self.data, mujoco.MjvOption(), 
                None, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
            )
            if render:
                mujoco.mjr_render(viewport, self.scene, self.context)

                # Swap OpenGL buffers
                glfw.swap_buffers(self.window)
                glfw.poll_events()
                time.sleep(0.001)
        print("RUN DONE")

def coords_to_angles(x, y, z, qw, qx, qy, qz):
    """
    Calculate joint angles for the robot arm to reach a target position and orientation
    using a custom inverse kinematics solution without external libraries.
    
    Args:
        x, y, z: Target position coordinates
        qw, qx, qy, qz: Target orientation as a quaternion
    
    Returns:
        Tuple of 6 joint angles (j1, j2, j3, j4, j5, j6)
    """
    print(f"Calculating IK for position ({x}, {y}, {z}) with orientation quaternion ({qw}, {qx}, {qy}, {qz})")
    
    # Robot dimensions from MuJoCo model (based on the XML)
    l1 = 1.0    # Length of first arm segment (from j2 to j3)
    l2 = 1.0    # Length of second arm segment (from j3 to j4)
    l3 = 0.2    # Length of third arm segment (from j4 to j5)
    l4 = 0.17   # Length of fourth arm segment (from j5 to j6)
    l5 = 0.2    # Length of end effector (from j6 to end)
    
    # Convert quaternion to rotation matrix
    rot = R.from_quat([qx, qy, qz, qw])
    rot_matrix = rot.as_matrix()
    
    # Extract the orientation axes
    x_axis = rot_matrix[:, 0]  # Forward direction of end effector
    y_axis = rot_matrix[:, 1]  # Left direction of end effector
    z_axis = rot_matrix[:, 2]  # Up direction of end effector
    
    # Wrist position: move back from target along x-axis
    wrist_pos = np.array([x, y, z]) - l5 * x_axis
    
    # Step 1: Calculate j1 (base rotation)
    j1 = math.atan2(wrist_pos[1], wrist_pos[0])
    
    # Distance from base to wrist in XY plane
    r_xy = math.sqrt(wrist_pos[0]**2 + wrist_pos[1]**2)
    
    # Height of wrist above base
    h = wrist_pos[2]
    
    # We need to solve for j2 and j3
    # The arm's kinematic structure from the XML: 
    # - j2 rotates around -Y at the base
    # - j3 rotates around -Y at the end of the first segment
    
    # Simplified 2D problem: reach [r_xy, h] with two segments
    # Length of segments in the 2D problem
    seg1 = l1
    seg2 = l2 + l3 + l4  # Combined length of remaining segments
    
    # Distance from base to wrist
    d = math.sqrt(r_xy**2 + h**2)
    
    # Angle from horizontal to wrist
    phi = math.atan2(h, r_xy)
    
    # Check if target is reachable
    if d > (seg1 + seg2):
        print(f"Warning: Target position is out of reach. Distance: {d:.3f}, Max reach: {seg1 + seg2:.3f}")
        # Scale the position to be reachable
        wrist_pos = wrist_pos * (0.95 * (seg1 + seg2) / d)
        r_xy = math.sqrt(wrist_pos[0]**2 + wrist_pos[1]**2)
        h = wrist_pos[2]
        d = math.sqrt(r_xy**2 + h**2)
        phi = math.atan2(h, r_xy)
    
    # Calculate j3 using law of cosines
    cos_j3 = (d**2 - seg1**2 - seg2**2) / (2 * seg1 * seg2)
    
    # Ensure cos_j3 is in valid range [-1, 1]
    cos_j3 = max(-1.0, min(1.0, cos_j3))
    
    # Elbow angle (negative due to rotation around -Y)
    j3 = -math.acos(cos_j3)
    
    # Calculate j2 using geometry
    # Angle from segment 1 to the line from base to wrist
    beta = math.acos((seg1**2 + d**2 - seg2**2) / (2 * seg1 * d))
    
    # Shoulder angle
    j2 = phi - beta
    
    # Now for the wrist angles (j4, j5, j6)
    # We need to determine orientation of the end effector
    
    # Calculate the rotation matrix from base to the end of j3 (before wrist)
    c1, s1 = math.cos(j1), math.sin(j1)
    c2, s2 = math.cos(j2), math.sin(j2)
    c3, s3 = math.cos(j3), math.sin(j3)
    
    # Rotation matrices for each joint
    R1 = np.array([
        [c1, -s1, 0],
        [s1, c1, 0],
        [0, 0, 1]
    ])
    
    R2 = np.array([
        [c2, 0, s2],
        [0, 1, 0],
        [-s2, 0, c2]
    ])
    
    R3 = np.array([
        [c3, 0, s3],
        [0, 1, 0],
        [-s3, 0, c3]
    ])
    
    # Combined rotation matrix up to j3
    R_0_3 = R1 @ R2 @ R3
    
    # Desired rotation from j3 to end effector
    R_3_6 = R_0_3.T @ rot_matrix
    
    # Extract Euler angles ZYX from this rotation matrix
    # This gives us j4, j5, j6
    try:
        # Check for singularity
        if abs(R_3_6[0, 2]) >= 0.99999:
            # Gimbal lock case
            j5 = math.pi/2 if R_3_6[0, 2] > 0 else -math.pi/2
            j4 = 0  # Can be arbitrary
            j6 = math.atan2(R_3_6[1, 0], R_3_6[1, 1])
        else:
            j5 = math.asin(-R_3_6[0, 2])
            j4 = math.atan2(R_3_6[1, 2], R_3_6[2, 2])
            j6 = math.atan2(R_3_6[0, 1], R_3_6[0, 0])
    except Exception as e:
        print(f"Error calculating wrist angles: {e}")
        # Fallback to simple values
        j4 = j5 = j6 = 0
    
    # In our robot model, j4 and j5 rotate around -Y, so we need to negate them
    j4 = -j4
    j5 = -j5
    
    # Normalize all angles to the range [-π, π]
    j1 = ((j1 + math.pi) % (2 * math.pi)) - math.pi
    j2 = ((j2 + math.pi) % (2 * math.pi)) - math.pi
    j3 = ((j3 + math.pi) % (2 * math.pi)) - math.pi
    j4 = ((j4 + math.pi) % (2 * math.pi)) - math.pi
    j5 = ((j5 + math.pi) % (2 * math.pi)) - math.pi
    j6 = ((j6 + math.pi) % (2 * math.pi)) - math.pi
    
    print(f"Computed joint angles: j1={j1:.4f}, j2={j2:.4f}, j3={j3:.4f}, j4={j4:.4f}, j5={j5:.4f}, j6={j6:.4f}")
    
    return j1, j2, j3, j4, j5, j6

def calc_error():
    """
    Calculate the Euclidean distance between the target position (cursor) 
    and the robot end effector.
    """
    cursor = scene.data.body("cursor").xpos
    endpt = scene.data.body("endpt").xpos
    tot = 0
    for i in range(3):
        tot += (cursor[i] - endpt[i])**2
    return tot**.5

def main():
    global scene
    scene = InteractiveScene()
    j1 = j2 = j3 = j4 = j5 = j6 = x = y = z = qw = qx = qy = qz = 0
    while True:
        steps = 3000
        try:
            x_input = input("coordinates and rotation (x y z qw qx qy qz): ")
            if not x_input.strip():
                continue
                
            if x_input:
                inp = x_input.strip().split()
                if len(inp) != 7:
                    print("Error: Please enter 7 values (x y z qw qx qy qz)")
                    continue
                    
                try:
                    x, y, z, qw, qx, qy, qz = [float(val) for val in inp]
                except ValueError:
                    print("Error: All values must be numbers")
                    continue
                    
                # Normalize quaternion
                quat_norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
                if abs(quat_norm - 1.0) > 0.01:
                    print(f"Warning: Quaternion not normalized. Norm = {quat_norm}")
                    qw /= quat_norm
                    qx /= quat_norm
                    qy /= quat_norm
                    qz /= quat_norm
                    print(f"Normalized quaternion: {qw:.4f} {qx:.4f} {qy:.4f} {qz:.4f}")
                
                print(f"Moving cursor to position ({x}, {y}, {z})")
                scene.run(steps, j1, j2, j3, j4, j5, j6, x, y, z, qw, qx, qy, qz)
                print("Calculating joint angles...")
                j1, j2, j3, j4, j5, j6 = coords_to_angles(x, y, z, qw, qx, qy, qz)
                print(f"Calculated joint angles: {j1:.4f}, {j2:.4f}, {j3:.4f}, {j4:.4f}, {j5:.4f}, {j6:.4f}")
                
            scene.run(steps, j1, j2, j3, j4, j5, j6)
            error = calc_error()
            print(f"ERROR: {error:.6f}")
            if error < 0.1:
                print("Target position reached successfully!")
            elif error < 0.5:
                print("Close to target position.")
            else:
                print("Warning: Significant error in position. Target may be unreachable or IK solution is not optimal.")
                
        except KeyboardInterrupt:
            print("\nExiting program...")
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--skip", type=int)
    args = parser.parse_args()

    main()
