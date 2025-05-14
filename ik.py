from copy import copy
import random
import mujoco
import numpy as np
import glfw
import argparse
import time
import sys, select

MAXDIST = 2

# f = open("claude3d.xml")
f = open("tycho_arm.mjcf")
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


#Note this function has side effects (changes joint positions in scene)
def forward_kinematic(j1, j2, j3, j4, j5, j6):
    scene.data.joint('j1').qpos[0] = j1
    scene.data.joint('j2').qpos[0] = j2
    scene.data.joint('j3').qpos[0] = j3
    scene.data.joint('j4').qpos[0] = j4
    scene.data.joint('j5').qpos[0] = j5
    scene.data.joint('j6').qpos[0] = j6
    scene.data.actuator('j1').ctrl[0] = j1
    scene.data.actuator('j2').ctrl[0] = j2
    scene.data.actuator('j3').ctrl[0] = j3
    scene.data.actuator('j4').ctrl[0] = j4
    scene.data.actuator('j5').ctrl[0] = j5
    scene.data.actuator('j6').ctrl[0] = j6
    scene.run(1)
    
def calc_error():
    curpos = scene.data.body("cursor").xpos
    curq = scene.data.body("cursor").xquat
    endpt = scene.data.body("endpt").xpos
    endq = scene.data.body("endpt").xquat
    tot = 0
    for i in range(4):
        tot += (curq[i]-endq[i])**2 * .1
    q = tot
    for i in range(3):
        tot += (curpos[i]-endpt[i])**2
    err = tot**.5
    print ("cur:", curpos, "end:", endpt, "rot:", q**.5, "total:", err)
    return err

def coords_to_angles(x, y, z, qw, qx, qy, qz):
    best = 999999
    delta = 0.1
    winner = [0, 0, 0, 0, 0, 0]
    for i in range(1000):
        ang_old = copy(winner)
        ang_new = []
        maxx = 2.875
        for j in range(6):
            ang_new.append(ang_old[j] + (random.gauss(0, delta)))
            if j == 5:
                maxx = 3.14
            if ang_new[j] < -maxx:
                ang_new[j] = -maxx
            if ang_new[j] > maxx:
                ang_new[j] = maxx
        forward_kinematic(*ang_new)
        err = calc_error()
        if err < best:
            best = err
            winner = copy(ang_new)
        print (i, "WINNER:", winner, best)
        # input()
        delta *= .997
    forward_kinematic(*winner)
    return winner

def main():
    global scene, j1, j2, j3, j4, j5, j6
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
        print (f"IK ANGLES: {j1}, {j2}, {j3}, {j4}, {j5}, {j6}") 
        print (f"ACTUAL: {scene.data.joint('j1').qpos[0]}, {scene.data.joint('j2').qpos[0]}, {scene.data.joint('j3').qpos[0]}, {scene.data.joint('j4').qpos[0]}, {scene.data.joint('j5').qpos[0]}, {scene.data.joint('j6').qpos[0]}")
        scene.run(1, j1, j2, j3, j4, j5, j6)
        print (f"ACTUAL: {scene.data.joint('j1').qpos[0]}, {scene.data.joint('j2').qpos[0]}, {scene.data.joint('j3').qpos[0]}, {scene.data.joint('j4').qpos[0]}, {scene.data.joint('j5').qpos[0]}, {scene.data.joint('j6').qpos[0]}")
        input()
        scene.run(steps, j1, j2, j3, j4, j5, j6)
        print (f"ACTUAL: {scene.data.joint('j1').qpos[0]}, {scene.data.joint('j2').qpos[0]}, {scene.data.joint('j3').qpos[0]}, {scene.data.joint('j4').qpos[0]}, {scene.data.joint('j5').qpos[0]}, {scene.data.joint('j6').qpos[0]}")
        print ("ERROR:", calc_error())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--skip", type=int)
    args = parser.parse_args()

    main()
