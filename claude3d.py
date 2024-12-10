import mujoco
import numpy as np
import glfw
import time

# Define the XML string for the MuJoCo model
MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <light cutoff="100" ambient="0.25 0.25 0.25" diffuse=".7 .7 .7" specular=".85 .85 .85" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" castshadow="false"/>
        <!-- Ground plane -->
        <geom type="plane" size="5 5 0.1" rgba=".49 .49 .49 1"/>

        <!-- First object (fixed base) -->
        <body pos="0 0 1" name="object1">
            <joint type="free"/>
            <geom type="box" size=".1 .3 .1" rgba="1 0 0 1"/>

            <!-- Second object connected by hinge -->
            <body pos="0.2 0 0" name="object2">
                <joint type="hinge" axis="0 1 0"/>
                <geom type="box" size=".1 .1 .3" rgba="0 1 0 1"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""

def init_glfw():
    """Initialize GLFW and create a window"""
    if not glfw.init():
        return None
    window = glfw.create_window(1200, 900, "MuJoCo Demo", None, None)
    if not window:
        glfw.terminate()
        return None
    glfw.make_context_current(window)
    return window

def main():
    # Initialize GLFW and create window
    window = init_glfw()
    if window is None:
        return

    # Load the model from the XML string
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    # Create camera
    cam = mujoco.MjvCamera()
    cam.distance = 3.0
    cam.azimuth = 90.0
    cam.elevation = -45.0

    # Create scene and context
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    # Simulation loop
    while not glfw.window_should_close(window):
        time_prev = data.time

        while (data.time - time_prev < 1.0/60.0):
            mujoco.mj_step(model, data)

        # Get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        # Update scene and render
        mujoco.mj_forward(model, data)
        mujoco.mjv_updateScene(
            model, data, mujoco.MjvOption(), 
            None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene
        )
        mujoco.mjr_render(viewport, scene, context)

        # Swap OpenGL buffers
        glfw.swap_buffers(window)

        # Process pending GUI events
        glfw.poll_events()

        # Add small delay
        time.sleep(0.001)

    # Clean up
    glfw.terminate()

if __name__ == "__main__":
    main()
