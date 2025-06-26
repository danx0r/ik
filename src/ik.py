import math
import mujoco
import numpy as np
import glfw
import argparse
import time
import sys, select

RAD2DEG=57.2958
MAXDIST = 2
LINK_LENGTH1 = 1
LINK_LENGTH2 = 0.8

f = open("simple_arm.mjcf")
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
        glfw.set_key_callback(self.window, self._keyboard)

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

    def _keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS or act == glfw.REPEAT:
            # Camera rotation
            if key == glfw.KEY_LEFT:
                self.cam.azimuth += self.rotation_speed
            elif key == glfw.KEY_RIGHT:
                self.cam.azimuth -= self.rotation_speed
            elif key == glfw.KEY_UP:
                self.cam.elevation = min(self.cam.elevation + self.rotation_speed, 90)
            elif key == glfw.KEY_DOWN:
                self.cam.elevation = max(self.cam.elevation - self.rotation_speed, -90)
            
            # Camera zoom
            elif key == glfw.KEY_Z:
                self.cam.distance = max(self.cam.distance - self.zoom_speed, 0.5)
            elif key == glfw.KEY_X:
                self.cam.distance += self.zoom_speed
                
            # Camera panning
            elif key == glfw.KEY_W:
                self.cam.lookat[1] += self.pan_speed
            elif key == glfw.KEY_S:
                self.cam.lookat[1] -= self.pan_speed
            elif key == glfw.KEY_A:
                self.cam.lookat[0] -= self.pan_speed
            elif key == glfw.KEY_D:
                self.cam.lookat[0] += self.pan_speed
            elif key == glfw.KEY_Q:
                self.cam.lookat[2] += self.pan_speed
            elif key == glfw.KEY_E:
                self.cam.lookat[2] -= self.pan_speed
                
            # Reset camera
            elif key == glfw.KEY_SPACE:
                self.cam.distance = 3.0
                self.cam.azimuth = 90.0
                self.cam.elevation = -45.0
                self.cam.lookat[0] = 0
                self.cam.lookat[1] = 0
                self.cam.lookat[2] = 0

    def run(self, steps=999999, j1=None, j2=None, j3=None, j4=None, j5=None, j6=None, x=None, y=None, z=None, p=None, w=None, r=None, render=True):
        while steps > 0 and not glfw.window_should_close(self.window):
            if kbhit():
                input()
                break
            steps -= 1
            time_prev = self.data.time

            if x is not None:
                self.data.actuator('cursor_x').ctrl[0] = x
            if y is not None:
                self.data.actuator('cursor_y').ctrl[0] = y
            if z is not None:
                self.data.actuator('cursor_z').ctrl[0] = z
            if p is not None:
                self.data.actuator('cursor_p').ctrl[0] = p
            if w is not None:
                self.data.actuator('cursor_w').ctrl[0] = w
            if r is not None:
                self.data.actuator('cursor_r').ctrl[0] = r

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

def coords_to_angles(x, y, z, p, w=0, r=0, link1_length=LINK_LENGTH1, link2_length=LINK_LENGTH2):
    """
    Inverse kinematics for 5DOF arm with variable link lengths
    
    Args:
        x, y, z: Target position coordinates
        p: Pitch angle of end effector (radians)
        w: Yaw angle of end effector (radians) 
        r: Roll angle of end effector (radians)
        link1_length: Length of first link segment
        link2_length: Length of second link segment
    
    Returns:
        j1, j2, j3, j4, j5: Joint angles in degrees
    """
    # Calculate total reach and normalize coordinates
    max_reach = link1_length + link2_length
    
    # Calculate distances
    xy = (x**2 + y**2) ** 0.5
    xyz = (x**2 + y**2 + z**2) ** 0.5
    
    # Prevent division by zero
    if xy == 0:
        xy = 0.0000001
    if xyz == 0:
        xyz = 0.0000001
    
    # J1: Base rotation (yaw) - rotation around Z axis
    j1 = math.atan2(y, x) * RAD2DEG
    
    # Calculate position for the wrist center (subtract end effector offset)
    # Assuming the end effector extends some distance from the wrist
    # For now, we'll position the wrist at the target position
    wrist_x, wrist_y, wrist_z = x, y, z
    wrist_dist = (wrist_x**2 + wrist_y**2 + wrist_z**2) ** 0.5
    wrist_xy = (wrist_x**2 + wrist_y**2) ** 0.5
    
    # Check if target is reachable
    if wrist_dist > max_reach:
        # Target unreachable - extend arm fully toward target
        j3 = 0  # Fully extended elbow
        j2 = math.atan2(wrist_z, wrist_xy) * RAD2DEG
    else:
        # J3: Elbow angle using law of cosines
        cos_elbow = (link1_length**2 + link2_length**2 - wrist_dist**2) / (2 * link1_length * link2_length)
        cos_elbow = max(-1, min(1, cos_elbow))  # Clamp to valid range
        j3 = math.acos(cos_elbow) * RAD2DEG
        j3 = 180 - j3  # Convert to joint angle convention
        
        # J2: Shoulder pitch
        # Calculate angle from horizontal to target
        alpha = math.atan2(wrist_z, wrist_xy)
        # Calculate angle from link1 to target using law of cosines
        cos_shoulder = (link1_length**2 + wrist_dist**2 - link2_length**2) / (2 * link1_length * wrist_dist)
        cos_shoulder = max(-1, min(1, cos_shoulder))
        beta = math.acos(cos_shoulder)
        j2 = (alpha + beta) * RAD2DEG
    
    # J4, J5: End effector orientation (only 5 joints available)
    # These joints control the orientation of the end effector
    # J4: Wrist pitch (around Y axis) - combines pitch and yaw influence
    j4 = p * RAD2DEG + w * RAD2DEG * 0.5  # Blend pitch and yaw
    
    # J5: Wrist roll (around X axis)
    j5 = r * RAD2DEG
    
    return j1, j2, -j3, j4, j5

def calc_error():
    endpt = scene.data.body("endpt").xpos
    cursor = scene.data.body("cursor").xpos
    # print ("CALC_ERROR", endpt, cursor)
    tot = 0
    for i in range(3):
        tot += (endpt[i]-cursor[i])**2
    return tot**.5

def main():
    global scene
    scene = InteractiveScene()
    j1 = j2 = j3 = j4 = j5 = x = y = z = w = r = 0
    while True:
        steps = 3000000
        x = input("coordinates and rotation (x y z p w r): ")
        if x:
            inp = x.strip().split()
            if len(inp) >= 4:
                x, y, z, p = inp[0], inp[1], inp[2], inp[3]
                w = inp[4] if len(inp) > 4 else '0'
                r = inp[5] if len(inp) > 5 else '0'
                
                x = float(x)
                y = float(y)
                z = float(z)
                p = float(p)/RAD2DEG #cursor pitch
                w = float(w)/RAD2DEG #cursor yaw
                r = float(r)/RAD2DEG #cursor roll
                scene.run(steps/2, j1, j2, j3, j4, j5, x=x, y=y, z=z, p=p, w=w, r=r)

                j1, j2, j3, j4, j5 = coords_to_angles(x, y, z, p, w, r)
                print (f"ANGLES: {j1}, {j2}, {j3}, {j4}, {j5}")
                j1 = float(j1)/RAD2DEG
                j2 = float(j2)/RAD2DEG
                j3 = float(j3)/RAD2DEG
                j4 = float(j4)/RAD2DEG
                j5 = float(j5)/RAD2DEG

        scene.run(steps, j1, j2, j3, j4, j5)
        print ("ERROR:", calc_error())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--skip", type=int)
    args = parser.parse_args()

    main()
