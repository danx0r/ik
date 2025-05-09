import math
import mujoco
import numpy as np
import glfw
import argparse
import time
import sys, select
import ikpy.chain
from ikpy.link import Link
from scipy.spatial.transform import Rotation as R

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
    Calculate joint angles for the robot arm to reach a target position and orientation.
    
    Args:
        x, y, z: Target position coordinates
        qw, qx, qy, qz: Target orientation as a quaternion
    
    Returns:
        Tuple of 6 joint angles (j1, j2, j3, j4, j5, j6)
    """
    # From analyzing the MuJoCo XML model, we have a 6-DOF arm with the following structure:
    # j1: base rotation around Z axis
    # j2: shoulder joint around -Y axis
    # j3: elbow joint around -Y axis
    # j4: wrist pitch around -Y axis
    # j5: wrist roll around Z axis
    # j6: end effector rotation around X axis
    
    # Create the kinematic chain based on the MuJoCo model structure
    chain = ikpy.chain.Chain(name="arm", links=[
        # Base frame (fixed)
        Link(
            name="base",
            translation_vector=[0, 0, 0],
            orientation=[0, 0, 0],
            rotation=[0, 0, 0],
            bounds=None
        ),
        # Joint 1 (rotates around Z axis)
        Link(
            name="j1_link",
            translation_vector=[0, 0, 0],
            orientation=[0, 0, 0],
            rotation=[0, 0, 1],
            bounds=(-np.pi, np.pi)
        ),
        # Joint 2 (rotates around -Y axis)
        Link(
            name="j2_link",
            translation_vector=[0, 0, 0],
            orientation=[0, 0, 0],
            rotation=[0, -1, 0],
            bounds=(-np.pi, np.pi)
        ),
        # Joint 3 (rotates around -Y axis)
        Link(
            name="j3_link",
            translation_vector=[1, 0, 0],
            orientation=[0, 0, 0],
            rotation=[0, -1, 0],
            bounds=(-np.pi, np.pi)
        ),
        # Joint 4 (rotates around -Y axis)
        Link(
            name="j4_link",
            translation_vector=[1, 0, 0],
            orientation=[0, 0, 0],
            rotation=[0, -1, 0],
            bounds=(-np.pi, np.pi)
        ),
        # Joint 5 (rotates around Z axis)
        Link(
            name="j5_link",
            translation_vector=[0.2, 0, 0],
            orientation=[0, 0, 0],
            rotation=[0, 0, 1],
            bounds=(-np.pi, np.pi)
        ),
        # Joint 6 (rotates around X axis)
        Link(
            name="j6_link",
            translation_vector=[0.17, 0, 0],
            orientation=[0, 0, 0],
            rotation=[1, 0, 0],
            bounds=(-np.pi, np.pi)
        ),
        # End effector (fixed)
        Link(
            name="end_effector",
            translation_vector=[0.2, 0, 0],
            orientation=[0, 0, 0],
            rotation=[0, 0, 0],
            bounds=None
        )
    ])
    
    # Convert quaternion to rotation matrix
    # Note: ikpy expects rotation matrices in different format than scipy provides
    rot = R.from_quat([qx, qy, qz, qw])
    rot_matrix = rot.as_matrix()
    
    # Set up the target
    target_position = np.array([x, y, z])
    target_orientation = rot_matrix
    
    # Solve inverse kinematics
    # We'll use an initial position to help convergence
    initial_position = [0, 0, 0, 0, 0, 0, 0, 0]
    
    # Try first with orientation
    try:
        joint_angles = chain.inverse_kinematics(
            target_position=target_position,
            target_orientation=target_orientation,
            orientation_mode="matrix",
            initial_position=initial_position,
            max_iter=1000
        )
    except Exception as e:
        print(f"IK with orientation failed: {e}")
        # If that fails, try without orientation constraint
        try:
            joint_angles = chain.inverse_kinematics(
                target_position=target_position,
                initial_position=initial_position,
                max_iter=1000
            )
        except Exception as e:
            print(f"IK without orientation failed: {e}")
            # If all else fails, return a reasonable default
            return 0, 0, 0, 0, 0, 0
    
    # Extract the 6 joint angles (skipping the base and end effector)
    j1, j2, j3, j4, j5, j6 = joint_angles[1:7]
    
    # Normalize angles to the range [-π, π]
    j1 = ((j1 + np.pi) % (2 * np.pi)) - np.pi
    j2 = ((j2 + np.pi) % (2 * np.pi)) - np.pi
    j3 = ((j3 + np.pi) % (2 * np.pi)) - np.pi
    j4 = ((j4 + np.pi) % (2 * np.pi)) - np.pi
    j5 = ((j5 + np.pi) % (2 * np.pi)) - np.pi
    j6 = ((j6 + np.pi) % (2 * np.pi)) - np.pi
    
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
        x_input = input("coordinates and rotation: ")
        if x_input:
            inp = x_input.strip().split()
            x, y, z, qw, qx, qy, qz = inp
            x = float(x)
            y = float(y)
            z = float(z)
            qw = float(qw)
            qx = float(qx)
            qy = float(qy)
            qz = float(qz)
            scene.run(steps, j1, j2, j3, j4, j5, j6, x, y, z, qw, qx, qy, qz)
            j1, j2, j3, j4, j5, j6 = coords_to_angles(x, y, z, qw, qx, qy, qz)
        scene.run(steps, j1, j2, j3, j4, j5, j6)
        print("ERROR:", calc_error())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--skip", type=int)
    args = parser.parse_args()

    main()
