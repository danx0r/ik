import math
import mujoco
import numpy as np
import glfw
import argparse
import time
import sys, select

RAD2DEG=57.2958
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
            # if r is not None:
            #     self.data.actuator('cursor_r').ctrl[0] = r

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
                self.data.actuator('j5').ctrl[0] = j6

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

def coords_to_angles(x, y, z, pitch, yaw, roll):
    xy = (x**2 + y**2) ** 0.5
    xyz = (x**2 + y**2 + z**2) ** 0.5
    if xy==0:
        xy = 0.0000001
    if xyz==0:
        xyz = 0.0000001 #OH yes I did
    ang = math.acos(min(1, (xyz/MAXDIST))) * RAD2DEG
    return 0, 0, 0, 0, 0, 0

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
    j1 = j2 = j3 = j4 = j5 = j6 = x = y = z = 0
    while True:
        steps = 300
        x = input("coordinates and rotation: ")
        if x:
            poff = 0.2          #length of last segment ('twixt j4 & endpt)
            inp = x.strip().split()
            x, y, z, p, w, r = inp
            x = float(x)
            y = float(y)
            z = float(z)
            p = float(p)/RAD2DEG #pitch
            w = float(w)/RAD2DEG #yaW; y was already taken
            r = float(r)/RAD2DEG #roll
            scene.run(steps, j1, j2, j3, j4, j5, j6, x, y, z, p, w, r)
            j1, j2, j3, j4, j5, j6 = coords_to_angles(x, y, z, p, w, r)
        scene.run(steps, j1, j2, j3, j4, j5, j6)
        print ("ERROR:", calc_error())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--skip", type=int)
    args = parser.parse_args()

    main()
