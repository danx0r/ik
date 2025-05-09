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

def coords_to_angles_4dof(x, y, z, qw, qx, qy, qz):
    """
    4-DOF inverse kinematics that positions j5 at the target position.
    
    Args:
        x, y, z: Target position coordinates (where j5 should be)
        qw, qx, qy, qz: Target orientation as quaternion
        
    Returns:
        j1, j2, j3, j4: First four joint angles in radians
    """
    # Link lengths from the XML file
    link1_length = 1.0  # First arm segment (j2 to j3)
    link2_length = 1.0  # Second arm segment (j3 to j4)
    link3_length = 0.2  # Third arm segment (j4 to j5/target)
    
    # Step 1: Solve for j1 (base rotation)
    j1 = math.atan2(y, x)
    
    # Step 2: Calculate the target position in the frame after j1 rotation
    c1, s1 = math.cos(j1), math.sin(j1)
    
    # Target position in frame after j1 rotation (rotated to xz-plane)
    x_r = c1 * x + s1 * y
    z_r = z
    
    # Calculate the distance from the base to the target
    r = math.sqrt(x_r**2 + z_r**2)
    
    # Check if the target is reachable
    max_reach = link1_length + link2_length + link3_length
    if r > max_reach:
        scale = max_reach / r
        x_r *= scale
        z_r *= scale
        r = max_reach
        print(f"Warning: Target out of reach, scaled to maximum distance")
    
    # Step 3: Use a geometric approach to solve the 3-link manipulator
    
    # We can formulate this as a geometric problem in the xz-plane
    # Let's define coordinates: j2 is at origin (0,0), target is at (x_r, z_r)
    
    # Using the law of cosines to solve for j3
    # Distance from j2 to target
    d = r
    
    # Calculate j3
    cos_j3 = (d**2 - link1_length**2 - (link2_length + link3_length)**2) / (2 * link1_length * (link2_length + link3_length))
    cos_j3 = max(min(cos_j3, 1.0), -1.0)
    j3 = -math.acos(cos_j3)  # Negative due to joint direction
    
    # Calculate j2
    beta = math.atan2(z_r, x_r)
    cos_gamma = (link1_length**2 + d**2 - (link2_length + link3_length)**2) / (2 * link1_length * d)
    cos_gamma = max(min(cos_gamma, 1.0), -1.0)
    gamma = math.acos(cos_gamma)
    j2 = beta + gamma
    
    # Calculate j4 (needs to be 0 to keep link2 and link3 as a straight line)
    j4 = 0
    
    # Optional: Add debug output
    print(f"Target position (j5): ({x}, {y}, {z})")
    print(f"Target in rotated frame: ({x_r}, 0, {z_r})")
    print(f"Distance from j2 to target: {d}")
    print(f"Calculated joint angles: j1={math.degrees(j1):.2f}°, j2={math.degrees(j2):.2f}°, j3={math.degrees(j3):.2f}°, j4={math.degrees(j4):.2f}°")
    
    # Forward kinematics check to verify j5 position
    # Position of j3
    p3x = link1_length * math.cos(j2)
    p3z = link1_length * math.sin(j2)
    
    # Position of j4
    p4x = p3x + link2_length * math.cos(j2 + j3)
    p4z = p3z + link2_length * math.sin(j2 + j3)
    
    # Position of j5
    p5x = p4x + link3_length * math.cos(j2 + j3 + j4)
    p5z = p4z + link3_length * math.sin(j2 + j3 + j4)
    
    # Rotate back to world coordinates
    p5_global_x = c1 * p5x - s1 * 0
    p5_global_y = s1 * p5x + c1 * 0
    p5_global_z = p5z
    
    print(f"Forward kinematics check - j5 position: ({p5_global_x:.3f}, {p5_global_y:.3f}, {p5_global_z:.3f})")
    print(f"Error: {math.sqrt((p5_global_x-x)**2 + (p5_global_y-y)**2 + (p5_global_z-z)**2):.6f}")
    
    # Try another approach: solve for j2, j3, j4 separately
    # Let's find position of j4 first
    
    # We know the target is link3_length away from j4
    # Direction from j2 to target
    dir_to_target = np.array([x_r, z_r])
    dir_to_target_norm = dir_to_target / np.linalg.norm(dir_to_target)
    
    # j4 position (move back from target by link3_length)
    j4_pos_x = x_r - link3_length * dir_to_target_norm[0]
    j4_pos_z = z_r - link3_length * dir_to_target_norm[1]
    
    # Distance from j2 to j4
    r_j4 = math.sqrt(j4_pos_x**2 + j4_pos_z**2)
    
    # If j4 is not reachable with link1 and link2
    if r_j4 > link1_length + link2_length:
        scale = (link1_length + link2_length) / r_j4
        j4_pos_x *= scale
        j4_pos_z *= scale
        r_j4 = link1_length + link2_length
        print(f"Warning: j4 position not reachable, scaled")
    
    # Now solve for j2 and j3 to position j4
    cos_j3_new = (r_j4**2 - link1_length**2 - link2_length**2) / (2 * link1_length * link2_length)
    cos_j3_new = max(min(cos_j3_new, 1.0), -1.0)
    j3_new = -math.acos(cos_j3_new)
    
    beta_new = math.atan2(j4_pos_z, j4_pos_x)
    cos_gamma_new = (link1_length**2 + r_j4**2 - link2_length**2) / (2 * link1_length * r_j4)
    cos_gamma_new = max(min(cos_gamma_new, 1.0), -1.0)
    gamma_new = math.acos(cos_gamma_new)
    j2_new = beta_new + gamma_new
    
    # Calculate j4 to point from j4 to target
    # Direction of the arm at j4
    arm_dir_x = math.cos(j2_new + j3_new)
    arm_dir_z = math.sin(j2_new + j3_new)
    
    # Direction from j4 to target
    target_dir_x = x_r - j4_pos_x
    target_dir_z = z_r - j4_pos_z
    
    # Normalize
    target_dir_length = math.sqrt(target_dir_x**2 + target_dir_z**2)
    if target_dir_length > 1e-6:
        target_dir_x /= target_dir_length
        target_dir_z /= target_dir_length
    
    # Angle between arm direction and target direction
    cos_angle = arm_dir_x * target_dir_x + arm_dir_z * target_dir_z
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle = math.acos(cos_angle)
    
    # Determine the sign of the angle
    cross_product = arm_dir_x * target_dir_z - arm_dir_z * target_dir_x
    if cross_product < 0:
        angle = -angle
    
    j4_new = angle
    
    # Since j4 rotates around -y, negate the angle
    j4_new = -j4_new
    
    print(f"Alternative solution - j1={math.degrees(j1):.2f}°, j2={math.degrees(j2_new):.2f}°, j3={math.degrees(j3_new):.2f}°, j4={math.degrees(j4_new):.2f}°")
    
    # Forward kinematics check for alternative solution
    p3x_new = link1_length * math.cos(j2_new)
    p3z_new = link1_length * math.sin(j2_new)
    
    p4x_new = p3x_new + link2_length * math.cos(j2_new + j3_new)
    p4z_new = p3z_new + link2_length * math.sin(j2_new + j3_new)
    
    p5x_new = p4x_new + link3_length * math.cos(j2_new + j3_new + j4_new)
    p5z_new = p4z_new + link3_length * math.sin(j2_new + j3_new + j4_new)
    
    p5_global_x_new = c1 * p5x_new - s1 * 0
    p5_global_y_new = s1 * p5x_new + c1 * 0
    p5_global_z_new = p5z_new
    
    error_new = math.sqrt((p5_global_x_new-x)**2 + (p5_global_y_new-y)**2 + (p5_global_z_new-z)**2)
    print(f"Alternative solution - j5 position: ({p5_global_x_new:.3f}, {p5_global_y_new:.3f}, {p5_global_z_new:.3f})")
    print(f"Alternative solution - Error: {error_new:.6f}")
    
    # Use the approach with the smaller error
    if error_new < math.sqrt((p5_global_x-x)**2 + (p5_global_y-y)**2 + (p5_global_z-z)**2):
        j2, j3, j4 = j2_new, j3_new, j4_new
        print("Using alternative solution as it has smaller error")
    
    # Normalize angles to be within -pi to pi
    j1 = ((j1 + math.pi) % (2 * math.pi)) - math.pi
    j2 = ((j2 + math.pi) % (2 * math.pi)) - math.pi
    j3 = ((j3 + math.pi) % (2 * math.pi)) - math.pi
    j4 = ((j4 + math.pi) % (2 * math.pi)) - math.pi
    
    return j1, j2, j3, j4

def coords_to_angles(x, y, z, qw, qx, qy, qz):
    """
    Full inverse kinematics function for testing.
    For now, this calls the 4-DOF version and returns zeros for j5, j6
    
    Args:
        x, y, z: Target position coordinates
        qw, qx, qy, qz: Target orientation as quaternion
        
    Returns:
        j1, j2, j3, j4, j5, j6: Joint angles in radians
    """
    j1, j2, j3, j4 = coords_to_angles_4dof(x, y, z, qw, qx, qy, qz)
    j5 = j6 = 0  # We'll implement these later
    
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
