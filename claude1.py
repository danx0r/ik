import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

class RoboticArm2D:
    def __init__(self, l1, l2):
        """
        Initialize a 2-segment robotic arm
        l1: length of first segment
        l2: length of second segment
        """
        self.l1 = l1  # Length of first arm segment
        self.l2 = l2  # Length of second arm segment
        self.current_angles = (0, 0)  # Current joint angles
        self.target_x = 0
        self.target_y = 0
        
    def forward_kinematics(self, theta1, theta2):
        """
        Calculate end effector position given joint angles
        theta1: angle of first joint (from horizontal)
        theta2: angle of second joint (relative to first segment)
        """
        # Position of first joint
        x1 = self.l1 * np.cos(theta1)
        y1 = self.l1 * np.sin(theta1)
        
        # Position of end effector
        x2 = x1 + self.l2 * np.cos(theta1 + theta2)
        y2 = y1 + self.l2 * np.sin(theta1 + theta2)
        
        return (x1, y1), (x2, y2)
    
    def inverse_kinematics(self, target_x, target_y):
        """
        Calculate joint angles for a given target position
        Uses the law of cosines
        """
        # Distance to target
        d = np.sqrt(target_x**2 + target_y**2)
        
        # Check if target is reachable
        if d > (self.l1 + self.l2) or d < abs(self.l1 - self.l2):
            raise ValueError("Target position is not reachable")
        
        # Calculate theta2 using law of cosines
        cos_theta2 = (d**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        theta2 = np.arccos(np.clip(cos_theta2, -1.0, 1.0))
        
        # Calculate theta1
        beta = np.arctan2(target_y, target_x)
        gamma = np.arccos((self.l1**2 + d**2 - self.l2**2) / (2 * self.l1 * d))
        theta1 = beta - gamma
        
        return theta1, theta2

    def interpolate_angles(self, start_angles, end_angles, num_steps):
        """
        Create smooth interpolation between start and end angles
        """
        theta1_interp = np.linspace(start_angles[0], end_angles[0], num_steps)
        theta2_interp = np.linspace(start_angles[1], end_angles[1], num_steps)
        return list(zip(theta1_interp, theta2_interp))

    def setup_interactive_plot(self):
        """
        Set up the interactive plot with animation
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-self.l1 - self.l2, self.l1 + self.l2)
        self.ax.set_ylim(-self.l1 - self.l2, self.l1 + self.l2)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.set_title('Interactive 2D Robotic Arm\nClick anywhere to set target')
        
        # Initialize plot elements
        self.segment1, = self.ax.plot([], [], 'b-', linewidth=2, label='Segment 1')
        self.segment2, = self.ax.plot([], [], 'g-', linewidth=2, label='Segment 2')
        self.joint_base = self.ax.plot(0, 0, 'ko', markersize=10, label='Base')[0]
        self.joint_mid = self.ax.plot([], [], 'ko', markersize=10, label='Joint')[0]
        self.end_effector = self.ax.plot([], [], 'ro', markersize=10, label='End Effector')[0]
        self.target = self.ax.plot([], [], 'rx', markersize=10, label='Target')[0]
        
        # Add workspace visualization
        workspace = patches.Circle((0, 0), self.l1 + self.l2, fill=False, color='gray', linestyle='--', alpha=0.5)
        inner_workspace = patches.Circle((0, 0), abs(self.l1 - self.l2), fill=False, color='gray', linestyle='--', alpha=0.5)
        self.ax.add_patch(workspace)
        self.ax.add_patch(inner_workspace)
        
        self.ax.legend()
        
        # Connect mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        return self.fig, self.ax

    def on_click(self, event):
        """
        Handle mouse clicks for target selection
        """
        if event.inaxes != self.ax:
            return
        
        self.target_x = event.xdata
        self.target_y = event.ydata
        
        try:
            # Calculate new target angles
            target_angles = self.inverse_kinematics(self.target_x, self.target_y)
            
            # Create animation
            num_steps = 30
            angle_steps = self.interpolate_angles(self.current_angles, target_angles, num_steps)
            
            # Update animation
            self.anim = FuncAnimation(
                self.fig, self.update_animation, frames=angle_steps,
                interval=50, blit=True, repeat=False
            )
            
            # Update current angles
            self.current_angles = target_angles
            
            # Update target marker
            self.target.set_data([self.target_x], [self.target_y])
            
            plt.draw()
            
        except ValueError as e:
            print(f"Error: {e}")

    def update_animation(self, angles):
        """
        Update animation frame
        """
        theta1, theta2 = angles
        (x1, y1), (x2, y2) = self.forward_kinematics(theta1, theta2)
        
        # Update segments
        self.segment1.set_data([0, x1], [0, y1])
        self.segment2.set_data([x1, x2], [y1, y2])
        
        # Update joints
        self.joint_mid.set_data([x1], [y1])
        self.end_effector.set_data([x2], [y2])
        
        return self.segment1, self.segment2, self.joint_mid, self.end_effector

def main():
    # Create a robotic arm with segment lengths of 2 and 1.5 units
    arm = RoboticArm2D(2, 1.5)
    
    # Set up interactive plot
    fig, ax = arm.setup_interactive_plot()
    
    # Initial position
    (x1, y1), (x2, y2) = arm.forward_kinematics(0, 0)
    arm.segment1.set_data([0, x1], [0, y1])
    arm.segment2.set_data([x1, x2], [y1, y2])
    arm.joint_mid.set_data([x1], [y1])
    arm.end_effector.set_data([x2], [y2])
    
    plt.show()

if __name__ == "__main__":
    main()

