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

    def run(self, steps=999999, j1=None, j2=None, j3=None, x=None, y=None, z=None, p=None, w=None, r=None, render=True):
        # self.data.actuator('j4').ctrl[0] = 1.57
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

def build_lookup():
    lookup = []
    f = open("coords.csv")
    for row in f.readlines():
        row = row.strip().split(",")
        row.pop(4)
        row.pop(0)
        row = [float(x) for x in row]
        lookup.append(row)
        # break
    return lookup

def coords_to_angles(x, y, z):
    xy = (x**2 + y**2) ** 0.5
    xyz = (x**2 + y**2 + z**2) ** 0.5
    if xy==0:
        xy = 0.0000001
    if xyz==0:
        xyz = 0.0000001 #OH yes I did
    yaw = math.acos(x/xy) * RAD2DEG
    if y < 0:
        yaw = -yaw
    pitch = math.acos(xy/xyz) * RAD2DEG
    if z < 0:
        pitch = -pitch
    ang = math.acos(min(1, (xyz/MAXDIST))) * RAD2DEG
    return yaw, pitch+ang, -2*ang

def angles_to_coords(j1, j2, j3):
    for row in lookup:
        if row[0] == j1 and row[1] == j2 and row[2] == j3:
            return row[3:]
    print ("Angles not in lookup")
    return 0, 0, 0

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
    j1 = j2 = j3 = x = y = z = 0
    while True:
        steps = 3000000
        if input("press x for cursor, j for joints: ")[0] == 'x':
            x = input("coordinates and rotation: ")
            if x:
                inp = x.strip().split()
                if len(inp) == 6:
                    x, y, z, p, w, r = inp
                else:
                    x, y, z = inp
                    p, w, r = (0, 0, 0)
                x = float(x)
                y = float(y)
                z = float(z)
                p = float(p)/RAD2DEG #pitch
                w = float(w)/RAD2DEG #yaW; y was already taken
                r = float(r)/RAD2DEG #roll
                scene.run(steps/2, j1, j2, j3, x, y, z, p, w, r)

                j1, j2, j3 = coords_to_angles(x, y, z)
                print (f"ANGLES: {j1}, {j2}, {j3}")
                j1 = float(j1)/RAD2DEG
                j2 = float(j2)/RAD2DEG
                j3 = float(j3)/RAD2DEG
        else:
            j = input("joint angles: ")
            if j:
                j1, j2, j3 = j.strip().split()
                j1 = float(j1)
                j2 = float(j2)
                j3 = float(j3)
                x, y, z = angles_to_coords(j1, j2, j3)
                print (f"ANGLES: {j1}, {j2}, {j3} COORDS: {x}, {y}, {z}")
                j1 /= RAD2DEG
                j2 /= RAD2DEG
                j3 /= RAD2DEG

        scene.run(steps, j1, j2, j3, x, y, z)
        print ("ERROR:", calc_error())

def create_lookup(fn, render=False, skip=5):
    global scene
    scene = InteractiveScene()
    f = open(fn, 'w')
    for j1 in range(-90, 95, skip):
        for j2 in range(-90, 185, skip):
            for j3 in range(0, 185, skip):
                steps = 1000
                scene.run(steps, j1/57.2958, j2/57.2958, j3/57.2958, render=render)
                coords = scene.data.body("endpt").xpos
                print (f"angles, {j1}, {j2}, {j3}, coords, {coords[0]:.5f}, {coords[1]:.5f}, {coords[2]:.5f}", file=f)
                f.flush()
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create_lookup")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--skip", type=int)
    args = parser.parse_args()

    if args.create_lookup:
        create_lookup(args.create_lookup, args.render, args.skip)
    else:
        lookup = build_lookup()
        main()
