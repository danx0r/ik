import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox, Button

class RoboticArm3D:
    def __init__(self, l1, l2, l3):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.current_angles = np.array([0, np.pi/4, -np.pi/4])
        
    def forward_kinematics(self, theta1, theta2, theta3):
        """Calculate end effector position given joint angles"""
        # First joint (after vertical offset)
        x1, y1, z1 = 0, 0, self.l1
        
        # Second joint
        x2 = self.l2 * np.sin(theta2) * np.cos(theta1)
        y2 = self.l2 * np.sin(theta2) * np.sin(theta1)
        z2 = self.l1 + self.l2 * np.cos(theta2)
        
        # End effector
        x3 = x2 + self.l3 * np.sin(theta2 + theta3) * np.cos(theta1)
        y3 = y2 + self.l3 * np.sin(theta2 + theta3) * np.sin(theta1)
        z3 = z2 + self.l3 * np.cos(theta2 + theta3)
        
        return (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)

    def calculate_jacobian(self, theta1, theta2, theta3):
        """Calculate the Jacobian matrix"""
        # Get end effector position for current angles
        _, _, (x, y, z) = self.forward_kinematics(theta1, theta2, theta3)
        
        # Partial derivatives for theta1
        dx_dth1 = -y
        dy_dth1 = x
        dz_dth1 = 0
        
        # Partial derivatives for theta2
        dx_dth2 = self.l2 * np.cos(theta2) * np.cos(theta1) + self.l3 * np.cos(theta2 + theta3) * np.cos(theta1)
        dy_dth2 = self.l2 * np.cos(theta2) * np.sin(theta1) + self.l3 * np.cos(theta2 + theta3) * np.sin(theta1)
        dz_dth2 = -self.l2 * np.sin(theta2) - self.l3 * np.sin(theta2 + theta3)
        
        # Partial derivatives for theta3
        dx_dth3 = self.l3 * np.cos(theta2 + theta3) * np.cos(theta1)
        dy_dth3 = self.l3 * np.cos(theta2 + theta3) * np.sin(theta1)
        dz_dth3 = -self.l3 * np.sin(theta2 + theta3)
        
        return np.array([
            [dx_dth1, dx_dth2, dx_dth3],
            [dy_dth1, dy_dth2, dy_dth3],
            [dz_dth1, dz_dth2, dz_dth3]
        ])

    def inverse_kinematics_jacobian(self, target_x, target_y, target_z, max_iterations=1000, tolerance=1e-4):
        """Solve inverse kinematics using the Jacobian method"""
        angles = self.current_angles.copy()
        
        for i in range(max_iterations):
            # Get current end effector position
            _, _, (x, y, z) = self.forward_kinematics(*angles)
            
            # Calculate error
            error = np.array([target_x - x, target_y - y, target_z - z])
            error_magnitude = np.linalg.norm(error)
            
            if error_magnitude < tolerance:
                self.current_angles = angles
                return angles
            
            # Calculate Jacobian
            J = self.calculate_jacobian(*angles)
            
            # Use pseudoinverse for better stability
            J_inv = np.linalg.pinv(J)
            
            # Update angles
            delta_theta = J_inv @ error
            angles += delta_theta * 0.1  # Small step size for stability
            
            # Normalize angles to -pi to pi
            angles = np.mod(angles + np.pi, 2 * np.pi) - np.pi
            
        raise ValueError(f"Failed to converge. Final error: {error_magnitude}")

    def check_reachability(self, target_x, target_y, target_z):
        """Check if target is within reachable workspace"""
        d = np.sqrt(target_x**2 + target_y**2 + (target_z - self.l1)**2)
        if d > (self.l2 + self.l3) or d < abs(self.l2 - self.l3):
            raise ValueError(f"Target position ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}) is not reachable.\n"
                           f"Distance from first joint: {d:.2f}\n"
                           f"Maximum reach: {self.l2 + self.l3:.2f}\n"
                           f"Minimum reach: {abs(self.l2 - self.l3):.2f}")
    
    def setup_interactive_plot(self):
        self.fig = plt.figure(figsize=(15, 8))
        self.ax = self.fig.add_subplot(121, projection='3d')
        
        max_reach = self.l1 + self.l2 + self.l3
        self.ax.set_xlim(-max_reach, max_reach)
        self.ax.set_ylim(-max_reach, max_reach)
        self.ax.set_zlim(0, max_reach * 1.2)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Robotic Arm (3-DOF)')
        
        # Plot elements
        self.segment1, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Segment 1')
        self.segment2, = self.ax.plot([], [], [], 'g-', linewidth=2, label='Segment 2')
        self.segment3, = self.ax.plot([], [], [], 'r-', linewidth=2, label='Segment 3')
        
        self.joint_base = self.ax.scatter([0], [0], [0], color='black', s=100, label='Base')
        self.joint1 = self.ax.scatter([], [], [], color='black', s=100, label='Joint 1')
        self.joint2 = self.ax.scatter([], [], [], color='black', s=100, label='Joint 2')
        self.end_effector = self.ax.scatter([], [], [], color='red', s=100, label='End Effector')
        self.target_point = self.ax.scatter([], [], [], color='green', s=100, marker='*', label='Target')
        
        self.ax.legend()
        
        # Controls
        ax_x = plt.axes([0.65, 0.8, 0.2, 0.05])
        ax_y = plt.axes([0.65, 0.7, 0.2, 0.05])
        ax_z = plt.axes([0.65, 0.6, 0.2, 0.05])
        ax_button = plt.axes([0.65, 0.5, 0.2, 0.05])
        
        self.text_x = TextBox(ax_x, 'X:', initial='2.0')
        self.text_y = TextBox(ax_y, 'Y:', initial='2.0')
        self.text_z = TextBox(ax_z, 'Z:', initial='3.0')
        self.button = Button(ax_button, 'Move to Target')
        
        self.status_text = plt.figtext(0.65, 0.4, '', wrap=True)
        self.button.on_clicked(self.on_button_click)
        
        # Initial position
        self.update_plot(self.current_angles)
        
        return self.fig, self.ax
    
    def on_button_click(self, event):
        try:
            target_x = float(self.text_x.text)
            target_y = float(self.text_y.text)
            target_z = float(self.text_z.text)
            
            # Check reachability first
            self.check_reachability(target_x, target_y, target_z)
            
            # Update target visualization
            self.target_point._offsets3d = ([target_x], [target_y], [target_z])
            
            # Calculate inverse kinematics
            angles = self.inverse_kinematics_jacobian(target_x, target_y, target_z)
            
            # Update arm position
            self.update_plot(angles)
            
            # Verify position
            _, _, (ex, ey, ez) = self.forward_kinematics(*angles)
            error = np.sqrt((ex-target_x)**2 + (ey-target_y)**2 + (ez-target_z)**2)
            
            self.status_text.set_text(
                f'Target: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})\n'
                f'Achieved: ({ex:.2f}, {ey:.2f}, {ez:.2f})\n'
                f'Error: {error:.6f} units\n'
                f'Angles (degrees):\n'
                f'θ1: {np.degrees(angles[0]):.1f}°\n'
                f'θ2: {np.degrees(angles[1]):.1f}°\n'
                f'θ3: {np.degrees(angles[2]):.1f}°'
            )
            
        except ValueError as e:
            self.status_text.set_text(f'Error: {str(e)}')
        
        self.fig.canvas.draw_idle()
    
    def update_plot(self, angles):
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = self.forward_kinematics(*angles)
        
        self.segment1.set_data_3d([0, x1], [0, y1], [0, z1])
        self.segment2.set_data_3d([x1, x2], [y1, y2], [z1, z2])
        self.segment3.set_data_3d([x2, x3], [y2, y3], [z2, z3])
        
        self.joint1._offsets3d = ([x1], [y1], [z1])
        self.joint2._offsets3d = ([x2], [y2], [z2])
        self.end_effector._offsets3d = ([x3], [y3], [z3])

def main():
    arm = RoboticArm3D(1.5, 2.0, 1.5)
    fig, ax = arm.setup_interactive_plot()
    plt.show()

if __name__ == "__main__":
    main()
