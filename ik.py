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

def coords_to_angles_5dof(x, y, z, qw, qx, qy, qz):
    """
    5-DOF inverse kinematics that positions the target body (at j4) to match the cursor
    and uses j4 and j5 to better align with the desired orientation.
    
    Args:
        x, y, z: Target position coordinates
        qw, qx, qy, qz: Target orientation as quaternion
        
    Returns:
        j1, j2, j3, j4, j5: First five joint angles in radians
    """
    # Link lengths from the XML file
    link1_length = 1.0    # From j2 to j3
    link2_length = 1.0    # From j3 to j4 (target body)
    
    # Convert quaternion to rotation matrix (target orientation)
    target_rot = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    
    # Step 1: Solve for j1 (base rotation)
    j1 = math.atan2(y, x)
    
    # Step 2: Calculate the target position in the frame after j1 rotation
    c1, s1 = math.cos(j1), math.sin(j1)
    
    # Transform the target position to the local frame after j1
    target_local = np.array([
        c1 * x + s1 * y,
        -s1 * x + c1 * y,
        z
    ])
    
    x_r = target_local[0]
    z_r = target_local[2]
    
    # Step 3: Calculate the distance from the shoulder to the target
    r = math.sqrt(x_r**2 + z_r**2)
    
    # Check if the target is reachable
    max_reach = link1_length + link2_length
    if r > max_reach:
        scale = max_reach / r
        x_r *= scale
        z_r *= scale
        r = max_reach
        print(f"Warning: Target out of reach, scaled to maximum distance")
    
    # Step 4: Solve for j2 and j3 using the law of cosines
    # Calculate j3 (elbow joint)
    cos_j3 = (r**2 - link1_length**2 - link2_length**2) / (2 * link1_length * link2_length)
    cos_j3 = max(min(cos_j3, 1.0), -1.0)  # Clamp to valid range
    j3 = -math.acos(cos_j3)  # Negative due to joint direction
    
    # Calculate j2 (shoulder joint)
    beta = math.atan2(z_r, x_r)
    cos_gamma = (link1_length**2 + r**2 - link2_length**2) / (2 * link1_length * r)
    cos_gamma = max(min(cos_gamma, 1.0), -1.0)  # Clamp to valid range
    gamma = math.acos(cos_gamma)
    
    j2 = beta + gamma
    
    # Step 5: Calculate orientation after j1, j2, j3
    R1 = np.array([
        [c1, -s1, 0],
        [s1, c1, 0],
        [0, 0, 1]
    ])
    
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
    
    # Combined rotation after j3
    R123 = R1 @ R2 @ R3
    
    # Step 6: Calculate the desired rotation for j4 and j5
    # First, determine what rotation is needed after j3 to achieve the target orientation
    R_needed = R123.T @ target_rot
    
    # From the XML file:
    # j4 rotates around -y axis (pitch)
    # j5 rotates around z axis (yaw)
    
    # For a rotation sequence: first j4 (around -y), then j5 (around z)
    # We extract the angles using the appropriate sequence
    
    # Extract j4 (pitch around -y)
    # Using arctangent of x and z components
    j4 = math.atan2(-R_needed[2, 0], R_needed[0, 0])
    
    # Create rotation matrix for j4
    c4, s4 = math.cos(j4), math.sin(j4)
    R4 = np.array([
        [c4, 0, -s4],  # Note the negative sign for -y axis
        [0, 1, 0],
        [s4, 0, c4]    # Note the sign change due to -y axis
    ])
    
    # Calculate the remaining rotation needed after j4
    R_after_j4 = R4.T @ R_needed
    
    # Extract j5 (yaw around z axis)
    # Using arctangent of x and y components
    j5 = math.atan2(R_after_j4[1, 0], R_after_j4[0, 0])
    
    # Optional: Add debug output
    print(f"Target position: ({x}, {y}, {z})")
    print(f"Target in local frame after j1: ({x_r:.3f}, 0, {z_r:.3f})")
    print(f"Distance from base to target: {r:.3f}")
    print(f"Joint angles: j1={math.degrees(j1):.2f}°, j2={math.degrees(j2):.2f}°, j3={math.degrees(j3):.2f}°, j4={math.degrees(j4):.2f}°, j5={math.degrees(j5):.2f}°")
    
    # Forward kinematics check for position
    j3_x = link1_length * math.cos(j2)
    j3_z = link1_length * math.sin(j2)
    
    j4_x = j3_x + link2_length * math.cos(j2 + j3)
    j4_z = j3_z + link2_length * math.sin(j2 + j3)
    
    # Transform back to world coordinates
    j4_world_x = c1 * j4_x
    j4_world_y = s1 * j4_x
    j4_world_z = j4_z
    
    print(f"Forward kinematics - target body position: ({j4_world_x:.3f}, {j4_world_y:.3f}, {j4_world_z:.3f})")
    print(f"Position error: {math.sqrt((j4_world_x-x)**2 + (j4_world_y-y)**2 + (j4_world_z-z)**2):.6f}")
    
    # Calculate full orientation after j5 to check alignment with target
    c5, s5 = math.cos(j5), math.sin(j5)
    R5 = np.array([
        [c5, -s5, 0],
        [s5, c5, 0],
        [0, 0, 1]
    ])
    
    # Full rotation matrix up to j5
    R_full = R123 @ R4 @ R5
    
    # Calculate the angle between the arm's x-axis and the target's x-axis
    arm_x_axis = R_full[:, 0]
    target_x_axis = target_rot[:, 0]
    
    alignment = np.dot(arm_x_axis, target_x_axis)
    alignment_angle = math.acos(np.clip(alignment, -1.0, 1.0)) * 180 / math.pi
    
    print(f"Orientation alignment error (degrees): {alignment_angle:.2f}°")
    
    # Normalize angles to be within -pi to pi
    j1 = ((j1 + math.pi) % (2 * math.pi)) - math.pi
    j2 = ((j2 + math.pi) % (2 * math.pi)) - math.pi
    j3 = ((j3 + math.pi) % (2 * math.pi)) - math.pi
    j4 = ((j4 + math.pi) % (2 * math.pi)) - math.pi
    j5 = ((j5 + math.pi) % (2 * math.pi)) - math.pi
    
    return j1, j2, j3, j4, j5

def coords_to_angles(x, y, z, qw, qx, qy, qz):
    """
    Full inverse kinematics function.
    
    Args:
        x, y, z: Target position coordinates
        qw, qx, qy, qz: Target orientation as quaternion
        
    Returns:
        j1, j2, j3, j4, j5, j6: Joint angles in radians
    """
    j1, j2, j3, j4, j5 = coords_to_angles_5dof(x, y, z, qw, qx, qy, qz)
    j6 = 0  # We'll implement this later
    
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
