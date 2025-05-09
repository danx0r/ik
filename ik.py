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
    using MuJoCo's built-in inverse kinematics functionality.
    
    Args:
        x, y, z: Target position coordinates
        qw, qx, qy, qz: Target orientation as a quaternion
    
    Returns:
        Tuple of 6 joint angles (j1, j2, j3, j4, j5, j6)
    """
    print(f"Calculating IK for position ({x}, {y}, {z}) with orientation quaternion ({qw}, {qx}, {qy}, {qz})")
    
    # Create a temporary model and data instance to work with
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    
    # Get the end effector body id ('endpt' in the model)
    end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "endpt")
    if end_effector_id < 0:
        print("Warning: End effector 'endpt' not found in model, using target body instead")
        end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    
    # Get the joint indices for the 6 joints
    joint_ids = []
    for i in range(1, 7):
        joint_name = f"j{i}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name} not found in model")
        joint_ids.append(joint_id)
    
    print(f"End effector ID: {end_effector_id}, Joint IDs: {joint_ids}")
    
    # Set up the target position and orientation
    target_pos = np.array([x, y, z])
    target_quat = np.array([qw, qx, qy, qz])
    
    # Create arrays for the MuJoCo IK solver
    jpos_start = np.zeros(model.nv)  # Starting joint positions
    jpos = np.zeros(model.nv)  # Output joint positions
    
    # We'll try multiple starting configurations
    starting_configs = [
        [0, 0, 0, 0, 0, 0],              # Zero position
        [0, 0.5, -0.5, 0, 0, 0],         # Slightly bent arm
        [np.pi/2, 0, 0, 0, 0, 0],        # Rotated base
        [0, -0.5, 0.5, 0, 0, 0],         # Alternative bend
        [np.pi/4, 0.5, -0.3, 0, 0, 0],   # Mixed configuration
    ]
    
    best_error = float('inf')
    best_solution = None
    
    for config in starting_configs:
        # Set the initial joint positions
        for i, joint_id in enumerate(joint_ids):
            jpos_start[model.jnt_qposadr[joint_id]] = config[i]
        
        # Reset data
        mujoco.mj_resetData(model, data)
        data.qpos[:] = jpos_start
        
        # Forward kinematics to calculate initial state
        mujoco.mj_forward(model, data)
        
        # Set up the IK parameters
        ik_iter = 100  # Maximum number of iterations
        ik_tol = 1e-6  # Tolerance for convergence
        
        try:
            # MuJoCo's IK implementation works with the full position vector
            jpos[:] = jpos_start
            
            # Get body id for the end effector
            bodyid = end_effector_id
            
            # Create a copy of data to work with
            temp_data = mujoco.MjData(model)
            temp_data.qpos[:] = jpos_start
            
            # Forward kinematics to calculate initial state
            mujoco.mj_forward(model, temp_data)
            
            # Use MuJoCo's IK to solve for joint positions
            for _ in range(ik_iter):
                # Save the current joint positions
                old_jpos = jpos.copy()
                
                # Update data with current joint positions
                temp_data.qpos[:] = jpos
                mujoco.mj_forward(model, temp_data)
                
                # Get the current end effector position and orientation
                curr_pos = temp_data.xpos[bodyid].copy()
                curr_quat = temp_data.xquat[bodyid].copy()
                
                # Calculate position and orientation errors
                pos_err = target_pos - curr_pos
                
                # Convert quaternions to rotation matrices for comparison
                curr_rot = R.from_quat([curr_quat[1], curr_quat[2], curr_quat[3], curr_quat[0]])
                target_rot = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
                
                # Get the relative rotation between current and target
                rel_rot = target_rot * curr_rot.inv()
                axis_angle = rel_rot.as_rotvec()
                
                # Combine position and orientation errors
                error = np.concatenate([pos_err, axis_angle])
                
                # Compute Jacobian for the end effector
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, temp_data, jacp, jacr, bodyid)
                
                # Combine position and rotation Jacobians
                jac = np.vstack([jacp, jacr])
                
                # Extract the columns corresponding to the actuated joints
                jac_reduced = np.zeros((6, len(joint_ids)))
                for i, joint_id in enumerate(joint_ids):
                    dof_idx = model.jnt_dofadr[joint_id]
                    jac_reduced[:, i] = jac[:, dof_idx]
                
                # Solve the IK problem using pseudoinverse
                jac_pinv = np.linalg.pinv(jac_reduced)
                delta = jac_pinv @ error
                
                # Update joint positions
                for i, joint_id in enumerate(joint_ids):
                    jpos[model.jnt_qposadr[joint_id]] += delta[i]
                
                # Check for convergence
                if np.linalg.norm(delta) < ik_tol:
                    break
                
                # Check for lack of progress
                if np.linalg.norm(jpos - old_jpos) < ik_tol / 10:
                    break
            
            # Calculate final error
            temp_data.qpos[:] = jpos
            mujoco.mj_forward(model, temp_data)
            final_pos = temp_data.xpos[bodyid].copy()
            pos_error = np.linalg.norm(target_pos - final_pos)
            
            print(f"Starting config {config}: Final position error: {pos_error:.6f}")
            
            if pos_error < best_error:
                best_error = pos_error
                best_solution = np.array([jpos[model.jnt_qposadr[joint_id]] for joint_id in joint_ids])
                
        except Exception as e:
            print(f"Error during IK calculation with starting config {config}: {e}")
            continue
    
    if best_solution is None:
        print("Failed to find a valid IK solution. Returning zeros.")
        return 0, 0, 0, 0, 0, 0
    
    # Normalize angles to the range [-π, π]
    for i in range(len(best_solution)):
        best_solution[i] = ((best_solution[i] + np.pi) % (2 * np.pi)) - np.pi
    
    j1, j2, j3, j4, j5, j6 = best_solution
    
    print(f"Best IK solution: j1={j1:.4f}, j2={j2:.4f}, j3={j3:.4f}, j4={j4:.4f}, j5={j5:.4f}, j6={j6:.4f}")
    print(f"Final position error: {best_error:.6f}")
    
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
