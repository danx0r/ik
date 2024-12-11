import mujoco
import numpy as np
import glfw
import time

f = open("claude3d.xml")
MODEL_XML = f.read()
f.close()

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
        self.cam.distance = 3.0
        self.cam.azimuth = 90.0
        self.cam.elevation = -45.0
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

    def run(self):
        while not glfw.window_should_close(self.window):
            time_prev = self.data.time

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
            mujoco.mjr_render(viewport, self.scene, self.context)

            # Swap OpenGL buffers
            glfw.swap_buffers(self.window)
            glfw.poll_events()

            time.sleep(0.001)

        glfw.terminate()

def main():
    scene = InteractiveScene()
    scene.run()

if __name__ == "__main__":
    main()
