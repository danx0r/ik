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
    5-DOF inverse kinematics that positions the endpoint (the j6 origin) at the target position.
    
    Args:
        x, y, z: Target position coordinates (where endpoint should be)
        qw, qx, qy, qz: Target orientation as quaternion
        
    Returns:
        j1, j2, j3, j4, j5: First five joint angles in radians
    """
    # Link lengths from the XML file
    link1_length = 1.0    # First arm segment (j2 to j3)
    link2_length = 1.0    # Second arm segment (j3 to j4)
    link3_length = 0.2    # Third arm segment (j4 to j5)
    link4_length = 0.17   # Fourth segment (j5 to endpoint/j6)
    
    # Convert quaternion to rotation matrix for orientation
    target_rot = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    
    # Extract the target direction (x-axis of the target orientation)
    target_direction = target_rot[:, 0]
    
    # Step 1: Solve for j1 (base rotation)
    j1 = math.atan2(y, x)
    
    # Step 2: Calculate the target position in the frame after j1 rotation
    c1, s1 = math.cos(j1), math.sin(j1)
    R1 = np.array([
        [c1, -s1, 0],
        [s1, c1, 0],
        [0, 0, 1]
    ])
    
    # Target position in rotated frame (after j1)
    x_r = c1 * x + s1 * y
    z_r = z
    
    # Calculate j5 position by moving backward from target along target direction
    # Rotate target direction to the frame after j1
    target_dir_j1 = R1.T @ target_direction
    target_dir_j1_norm = target_dir_j1 / np.linalg.norm(target_dir_j1)
    
    # j5 position in rotated frame
    j5_pos_x = x_r - link4_length * target_dir_j1_norm[0]
    j5_pos_z = z_r - link4_length * target_dir_j1_norm[2]
    
    # Now calculate j4 position by moving again along target direction
    j4_pos_x = j5_pos_x - link3_length * target_dir_j1_norm[0]
    j4_pos_z = j5_pos_z - link3_length * target_dir_j1_norm[2]
    
    # Calculate the distance from j2 to j4
    r_j4 = math.sqrt(j4_pos_x**2 + j4_pos_z**2)
    
    # Check if j4 position is reachable with link1 and link2
    max_reach_j4 = link1_length + link2_length
    if r_j4 > max_reach_j4:
        scale = max_reach_j4 / r_j4
        j4_pos_x *= scale
        j4_pos_z *= scale
        r_j4 = max_reach_j4
        print(f"Warning: j4 position not reachable, scaled")
    
    # Step 3: Solve for j2 and j3 to position j4
    cos_j3 = (r_j4**2 - link1_length**2 - link2_length**2) / (2 * link1_length * link2_length)
    cos_j3 = max(min(cos_j3, 1.0), -1.0)
    j3 = -math.acos(cos_j3)  # Negative due to joint direction
    
    beta = math.atan2(j4_pos_z, j4_pos_x)
    cos_gamma = (link1_length**2 + r_j4**2 - link2_length**2) / (2 * link1_length * r_j4)
    cos_gamma = max(min(cos_gamma, 1.0), -1.0)
    gamma = math.acos(cos_gamma)
    j2 = beta + gamma
    
    # Step 4: Calculate orientation of the arm at j4
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
    
    # Combined rotation up to j3
    R123 = R1 @ R2 @ R3
    
    # Direction of the arm at j4 (before j4 rotation)
    arm_dir_at_j4 = R123[:, 0]
    
    # Step 5: Calculate j4 to align with the target direction
    # Calculate the angle between current arm direction and target direction
    dot_product = np.clip(np.dot(arm_dir_at_j4, target_direction), -1.0, 1.0)
    angle = math.acos(dot_product)
    
    # Determine the sign of the angle
    cross_product = np.cross(arm_dir_at_j4, target_direction)
    if np.dot(cross_product, R123[:, 1]) < 0:  # Check against y-axis of arm
        angle = -angle
    
    # Since j4 rotates around -y in the model, negate the angle
    j4 = -angle
    
    # Step 6: Calculate j5 to handle any remaining rotation
    # After j4, the arm's x-axis should be aligned with target direction
    c4, s4 = math.cos(j4), math.sin(j4)
    R4 = np.array([
        [c4, 0, s4],
        [0, 1, 0],
        [-s4, 0, c4]
    ])
    
    # Combined rotation up to j4
    R1234 = R123 @ R4
    
    # The arm's z-axis after j4
    arm_z_after_j4 = R1234[:, 2]
    
    # The target's z-axis
    target_z = target_rot[:, 2]
    
    # Calculate the angle between the arm's z-axis and target's z-axis
    # Project to the plane perpendicular to the arm's x-axis (which is aligned with target direction)
    
    # Remove components along the arm's x-axis
    arm_z_projected = arm_z_after_j4 - np.dot(arm_z_after_j4, R1234[:, 0]) * R1234[:, 0]
    target_z_projected = target_z - np.dot(target_z, R1234[:, 0]) * R1234[:, 0]
    
    # Normalize
    if np.linalg.norm(arm_z_projected) > 1e-6 and np.linalg.norm(target_z_projected) > 1e-6:
        arm_z_projected = arm_z_projected / np.linalg.norm(arm_z_projected)
        target_z_projected = target_z_projected / np.linalg.norm(target_z_projected)
        
        # Calculate the angle
        dot = np.clip(np.dot(arm_z_projected, target_z_projected), -1.0, 1.0)
        j5 = math.acos(dot)
        
        # Determine the sign
        cross = np.cross(arm_z_projected, target_z_projected)
        if np.dot(cross, R1234[:, 0]) < 0:
            j5 = -j5
    else:
        j5 = 0
    
    # Optional: Add debug output
    print(f"Target position (endpoint): ({x}, {y}, {z})")
    print(f"Calculated j5 position: ({j5_pos_x*c1}, {j5_pos_x*s1}, {j5_pos_z})")
    print(f"Calculated j4 position: ({j4_pos_x*c1}, {j4_pos_x*s1}, {j4_pos_z})")
    print(f"Distance from j2 to j4: {r_j4}")
    print(f"Calculated joint angles: j1={math.degrees(j1):.2f}°, j2={math.degrees(j2):.2f}°, j3={math.degrees(j3):.2f}°, j4={math.degrees(j4):.2f}°, j5={math.degrees(j5):.2f}°")
    
    # Forward kinematics check
    p3x = link1_length * math.cos(j2)
    p3z = link1_length * math.sin(j2)
    
    p4x = p3x + link2_length * math.cos(j2 + j3)
    p4z = p3z + link2_length * math.sin(j2 + j3)
    
    # Local x-axis direction after j4
    c4, s4 = math.cos(j4), math.sin(j4)
    local_x_dir_x = math.cos(j2 + j3 + j4)
    local_x_dir_z = math.sin(j2 + j3 + j4)
    
    # Calculate j5 position
    p5x = p4x + link3_length * local_x_dir_x
    p5z = p4z + link3_length * local_x_dir_z
    
    # Local x-axis direction after j5
    c5, s5 = math.cos(j5), math.sin(j5)
    R5 = np.array([
        [1, 0, 0],
        [0, c5, -s5],
        [0, s5, c5]
    ])
    R12345 = R1234 @ R5
    local_dir_after_j5 = R12345[:, 0]
    
    # Calculate endpoint position
    endpoint_x = p5x + link4_length * local_dir_after_j5[0]
    endpoint_z = p5z + link4_length * local_dir_after_j5[2]
    
    # Rotate back to world coordinates
    endpoint_global_x = c1 * endpoint_x - s1 * 0
    endpoint_global_y = s1 * endpoint_x + c1 * 0
    endpoint_global_z = endpoint_z
    
    error = math.sqrt((endpoint_global_x-x)**2 + (endpoint_global_y-y)**2 + (endpoint_global_z-z)**2)
    print(f"Forward kinematics check - endpoint position: ({endpoint_global_x:.3f}, {endpoint_global_y:.3f}, {endpoint_global_z:.3f})")
    print(f"Error: {error:.6f}")
    
    # Normalize angles to be within -pi to pi
    j1 = ((j1 + math.pi) % (2 * math.pi)) - math.pi
    j2 = ((j2 + math.pi) % (2 * math.pi)) - math.pi
    j3 = ((j3 + math.pi) % (2 * math.pi)) - math.pi
    j4 = ((j4 + math.pi) % (2 * math.pi)) - math.pi
    j5 = ((j5 + math.pi) % (2 * math.pi)) - math.pi
    
    return j1, j2, j3, j4, j5

def coords_to_angles(x, y, z, qw, qx, qy, qz):
    """
    Full inverse kinematics function that positions the endpoint at the target location
    and orients it according to the quaternion.
    
    Args:
        x, y, z: Target position coordinates
        qw, qx, qy, qz: Target orientation as quaternion
        
    Returns:
        j1, j2, j3, j4, j5, j6: Joint angles in radians
    """
    j1, j2, j3, j4, j5 = coords_to_angles_5dof(x, y, z, qw, qx, qy, qz)
    
    # Calculate j6 to account for the final roll orientation
    # Convert quaternion to rotation matrix
    target_rot = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    
    # Calculate the orientation of the arm after j5
    c1, s1 = math.cos(j1), math.sin(j1)
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
    
    c4, s4 = math.cos(j4), math.sin(j4)
    R4 = np.array([
        [c4, 0, s4],
        [0, 1, 0],
        [-s4, 0, c4]
    ])
    
    c5, s5 = math.cos(j5), math.sin(j5)
    R5 = np.array([
        [1, 0, 0],
        [0, c5, -s5],
        [0, s5, c5]
    ])
    
    # Combined rotation after j5
    R12345 = R1 @ R2 @ R3 @ R4 @ R5
    
    # We need to find the rotation around the x-axis (j6) to align with the target orientation
    # The residual rotation needed is:
    R_residual = R12345.T @ target_rot
    
    # For a rotation around x-axis, we extract the roll angle
    j6 = math.atan2(R_residual[2, 1], R_residual[1, 1])
    
    # Normalize j6 to be within -pi to pi
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
