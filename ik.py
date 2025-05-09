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
    Calculate joint angles for the robot arm to reach a target position and orientation.
    Optimized for the specific MuJoCo model based on the XML.
    
    Args:
        x, y, z: Target position coordinates
        qw, qx, qy, qz: Target orientation as a quaternion
    
    Returns:
        Tuple of 6 joint angles (j1, j2, j3, j4, j5, j6)
    """
    print(f"Calculating IK for position ({x}, {y}, {z}) with orientation quaternion ({qw}, {qx}, {qy}, {qz})")
    
    # Use numerical optimization to find joint angles
    # This is more robust than analytical solutions for complex robots
    
    # Convert quaternion to rotation matrix once
    from scipy.spatial.transform import Rotation
    target_rot = Rotation.from_quat([qx, qy, qz, qw])
    target_rot_matrix = target_rot.as_matrix()
    
    # Function to calculate forward kinematics
    def forward_kinematics(joint_angles):
        j1, j2, j3, j4, j5, j6 = joint_angles
        
        # Rotation matrices for each joint
        # j1 rotates around Z
        R1 = np.array([
            [np.cos(j1), -np.sin(j1), 0],
            [np.sin(j1), np.cos(j1), 0],
            [0, 0, 1]
        ])
        
        # j2 rotates around -Y
        R2 = np.array([
            [np.cos(j2), 0, np.sin(j2)],
            [0, 1, 0],
            [-np.sin(j2), 0, np.cos(j2)]
        ])
        
        # j3 rotates around -Y
        R3 = np.array([
            [np.cos(j3), 0, np.sin(j3)],
            [0, 1, 0],
            [-np.sin(j3), 0, np.cos(j3)]
        ])
        
        # j4 rotates around -Y
        R4 = np.array([
            [np.cos(j4), 0, np.sin(j4)],
            [0, 1, 0],
            [-np.sin(j4), 0, np.cos(j4)]
        ])
        
        # j5 rotates around Z
        R5 = np.array([
            [np.cos(j5), -np.sin(j5), 0],
            [np.sin(j5), np.cos(j5), 0],
            [0, 0, 1]
        ])
        
        # j6 rotates around X
        R6 = np.array([
            [1, 0, 0],
            [0, np.cos(j6), -np.sin(j6)],
            [0, np.sin(j6), np.cos(j6)]
        ])
        
        # Translation vectors for each joint
        # Base to j1 (no translation)
        T1 = np.array([0, 0, 0])
        
        # j1 to j2 (no translation)
        T2 = np.array([0, 0, 0])
        
        # j2 to j3 (1 unit in x direction)
        T3 = np.array([1, 0, 0])
        
        # j3 to j4 (1 unit in x direction)
        T4 = np.array([1, 0, 0])
        
        # j4 to j5 (0.2 unit in x direction)
        T5 = np.array([0.2, 0, 0])
        
        # j5 to j6 (0.17 unit in x direction)
        T6 = np.array([0.17, 0, 0])
        
        # j6 to end effector (0.2 unit in x direction)
        Te = np.array([0.2, 0, 0])
        
        # Compute the position and orientation of the end effector
        # Start with identity rotation and zero position
        R = np.eye(3)
        p = np.zeros(3)
        
        # Apply each transformation sequentially
        R = R @ R1
        p = p + T1
        
        p = p + R @ T2
        R = R @ R2
        
        p = p + R @ T3
        R = R @ R3
        
        p = p + R @ T4
        R = R @ R4
        
        p = p + R @ T5
        R = R @ R5
        
        p = p + R @ T6
        R = R @ R6
        
        p = p + R @ Te
        
        return p, R
    
    # Objective function for optimization
    def objective(joint_angles):
        # Calculate forward kinematics
        p, R = forward_kinematics(joint_angles)
        
        # Calculate position error
        target_pos = np.array([x, y, z])
        pos_error = np.linalg.norm(p - target_pos)
        
        # Calculate orientation error
        ori_error = np.linalg.norm(R - target_rot_matrix, 'fro')
        
        # Weighted sum of errors (position is more important than orientation)
        return pos_error + 0.1 * ori_error
    
    # Initial guess for optimization
    # We'll try multiple starting points to avoid local minima
    best_solution = None
    best_error = float('inf')
    
    # Starting configurations to try
    starting_configs = [
        [0, 0, 0, 0, 0, 0],              # Zero position
        [0, np.pi/4, -np.pi/4, 0, 0, 0], # Slightly bent arm
        [np.pi/2, 0, 0, 0, 0, 0],        # Rotated base
        [0, -np.pi/4, np.pi/4, 0, 0, 0]  # Alternative bend
    ]
    
    for initial_guess in starting_configs:
        # Use coordinate descent method for optimization
        current_guess = np.array(initial_guess)
        current_error = objective(current_guess)
        
        # Max iterations and step size
        max_iter = 200
        step_size = 0.1
        min_step = 0.01
        
        for _ in range(max_iter):
            improved = False
            
            # Try adjusting each joint angle
            for i in range(6):
                # Try increasing the angle
                current_guess[i] += step_size
                new_error = objective(current_guess)
                
                if new_error < current_error:
                    current_error = new_error
                    improved = True
                else:
                    # If not improved, try decreasing
                    current_guess[i] -= 2 * step_size
                    new_error = objective(current_guess)
                    
                    if new_error < current_error:
                        current_error = new_error
                        improved = True
                    else:
                        # If still not improved, revert
                        current_guess[i] += step_size
            
            # If no improvement, reduce step size
            if not improved:
                step_size *= 0.5
                
                # If step size is too small, stop
                if step_size < min_step:
                    break
        
        # Check if this solution is better than previous ones
        if current_error < best_error:
            best_error = current_error
            best_solution = current_guess
    
    # Normalize angles to [-π, π]
    for i in range(6):
        best_solution[i] = ((best_solution[i] + np.pi) % (2 * np.pi)) - np.pi
    
    j1, j2, j3, j4, j5, j6 = best_solution
    
    print(f"Computed joint angles: j1={j1:.4f}, j2={j2:.4f}, j3={j3:.4f}, j4={j4:.4f}, j5={j5:.4f}, j6={j6:.4f}")
    print(f"Final error: {best_error:.6f}")
    
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
