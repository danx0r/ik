import math
import mujoco
import numpy as np
import glfw
import argparse
import time
import sys, select

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
        # self.data.body("cursor").xpos[0]=1
        # print ("DBG", self.data.body("cursor").xpos)
        # self.data.mocap_pos[0]=[-.5, 0, 0]
        # self.data.mocap_quat[0]=[1, 0, -.4, 0]                          #pitch nose up 45 degrees
        # self.data.mocap_quat[0]=[1, 0, 0, -.4]                          #yaw 45 degrees clockwise (top view)
        # self.data.mocap_quat[0]=[1, .4, 0, 0]                           #roll 45 degrees clockwise/bank right
        # self.data.mocap_quat[0]=[1, 0, -.2, .1]                           #nose up 25 deg, yaw 13 deg left (a bit of roll innit)
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
        print ("RUN DONE")

def coords_to_angles(x, y, z, qw, qx, qy, qz):
    """
    Inverse kinematics function to calculate joint angles from end-effector position and orientation.
    
    Args:
        x, y, z: Target position coordinates
        qw, qx, qy, qz: Target orientation as quaternion
        
    Returns:
        j1, j2, j3, j4, j5, j6: Joint angles in radians
    """
    # Extract exact measurements from the XML file
    link1_length = 1.0    # First arm segment (from j2 to j3)
    link2_length = 1.0    # Second arm segment (from j3 to j4)
    link3_length = 0.2    # Third arm segment (from j4 to j5)
    link4_length = 0.17   # From j5 to j6
    end_effector = 0.2    # End effector extension (from j6 to end point)
    
    # The target position is where the end point should be
    # We need to work backwards through the kinematic chain
    
    # Target rotation matrix from quaternion
    target_rot = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    
    # The end effector's x-axis (forward direction)
    ee_x_axis = target_rot[:, 0]
    
    # Calculate position of joint j6 by moving backward from the end point
    j6_pos = np.array([x, y, z]) - end_effector * ee_x_axis
    
    # Calculate position of joint j5 by moving backward from j6
    # j5 is located link4_length distance back from j6 along the x-axis
    j5_pos = j6_pos - link4_length * ee_x_axis
    
    # Calculate position of joint j4
    # j4 is located link3_length distance back from j5 along the local x-axis before j5 rotation
    # Since we don't know j5 angle yet, we need to use the direction from j3 to j4
    # For simplicity, we'll assume j4 is link3_length directly behind j5 in the global frame
    # This is an approximation but will give us a starting point
    j4_pos = j5_pos - link3_length * ee_x_axis
    
    # Now we solve for j1, j2, j3 to position j4
    wx, wy, wz = j4_pos
    
    # Step 1: Solve for j1 (base rotation)
    j1 = math.atan2(wy, wx)
    
    # Calculate position in the frame after j1 rotation
    c1, s1 = math.cos(j1), math.sin(j1)
    R1 = np.array([
        [c1, -s1, 0],
        [s1, c1, 0],
        [0, 0, 1]
    ])
    
    # Position of j4 in the local frame after j1
    j4_local = R1.T @ np.array([wx, wy, wz])
    wx_l, wy_l, wz_l = j4_local
    
    # Distance from joint 2 to j4 in the XZ plane after j1 rotation
    r = math.sqrt(wx_l**2 + wz_l**2)
    
    # Check if target is reachable by the first 3 joints
    max_reach = link1_length + link2_length
    if r > max_reach:
        # Scale to maximum reachable distance
        scale = max_reach / r
        wx_l *= scale
        wz_l *= scale
        r = max_reach
    
    # Step 2: Solve for j2 and j3 using the law of cosines
    # Calculate j3 (elbow joint)
    cos_j3 = (r**2 - link1_length**2 - link2_length**2) / (2 * link1_length * link2_length)
    cos_j3 = max(min(cos_j3, 1.0), -1.0)  # Clamp to valid range
    j3 = -math.acos(cos_j3)  # Negative due to joint direction
    
    # Calculate j2 (shoulder joint)
    # Angle from x-axis to the line from j2 to j4
    beta = math.atan2(wz_l, wx_l)
    
    # Angle between the line from j2 to j3 and the line from j2 to j4
    cos_gamma = (link1_length**2 + r**2 - link2_length**2) / (2 * link1_length * r)
    cos_gamma = max(min(cos_gamma, 1.0), -1.0)
    gamma = math.acos(cos_gamma)
    
    j2 = beta + gamma
    
    # Step 3: Calculate the orientation after j3
    c2, s2 = math.cos(j2), math.sin(j2)
    R2 = np.array([
        [c2, 0, s2],
        [0, 1, 0],
        [-s2, 0, c2]
    ])
    
    c3, s3 = math.cos(j3), math.sin(j3)
    R3 = np.array([
        [c3, 0, s3],
        [0, 1, 0],
        [-s3, 0, c3]
    ])
    
    # Combined rotation matrix after j3
    R123 = R1 @ R2 @ R3
    
    # Step 4: Find the remaining rotation needed to achieve the target orientation
    # This will determine j4, j5, and j6
    R_needed = R123.T @ target_rot
    
    # Extract Euler angles from the rotation matrix
    # Based on the XML, j4 is around -y, j5 is around z, j6 is around x
    # We'll use a YZX sequence to extract angles
    
    # Extract j5 first (rotation around z)
    j5 = math.atan2(R_needed[1, 0], R_needed[0, 0])
    
    # Calculate cos(j5) and sin(j5)
    c5, s5 = math.cos(j5), math.sin(j5)
    
    # Extract j4 (rotation around -y)
    # Since j4 is around -y, we use a negative sign
    j4 = -math.atan2(-R_needed[2, 0], c5*R_needed[0, 0] + s5*R_needed[1, 0])
    
    # Extract j6 (rotation around x)
    j6 = math.atan2(s5*R_needed[0, 2] - c5*R_needed[1, 2], 
                   -s5*R_needed[0, 1] + c5*R_needed[1, 1])
    
    # Normalize angles to be within -pi to pi
    j1 = ((j1 + math.pi) % (2 * math.pi)) - math.pi
    j2 = ((j2 + math.pi) % (2 * math.pi)) - math.pi
    j3 = ((j3 + math.pi) % (2 * math.pi)) - math.pi
    j4 = ((j4 + math.pi) % (2 * math.pi)) - math.pi
    j5 = ((j5 + math.pi) % (2 * math.pi)) - math.pi
    j6 = ((j6 + math.pi) % (2 * math.pi)) - math.pi
    
    return j1, j2, j3, j4, j5, j6    
def calc_error():
    target = scene.data.body("target").xpos
    cursor = scene.data.body("cursor").xpos
    # print ("CALC_ERROR", endpt, cursor)
    tot = 0
    for i in range(3):
        tot += (target[i]-cursor[i])**2
    return tot**.5

def main():
    global scene
    scene = InteractiveScene()
    j1 = j2 = j3 = j4 = j5 = j6 = x = y = z = qw = qx = qy = qz = 0
    while True:
        steps = 3000
        x = input("coordinates and rotation: ")
        if x:
            inp = x.strip().split()
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
        print ("ERROR:", calc_error())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--skip", type=int)
    args = parser.parse_args()

    main()
