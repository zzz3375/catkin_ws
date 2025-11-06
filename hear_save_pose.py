import rospy
import signal
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_matrix

# Storage for pose data
positions = []  # [x, y, z]
orientations = []  # [qx, qy, qz, qw]
timestamps = []  # ROS time

def signal_handler(sig, frame):
    """Handle exit signal, save data and create plots"""
    if not positions:
        rospy.logwarn("No pose data collected, no files saved")
        sys.exit(0)
    
    # Convert to numpy arrays
    positions_np = np.array(positions)
    orientations_np = np.array(orientations)
    timestamps_np = np.array(timestamps)
    
    # Calculate roll, pitch, yaw from quaternions
    rpy_list = []
    for orientation in orientations:
        roll, pitch, yaw = euler_from_quaternion(orientation)
        rpy_list.append([roll, pitch, yaw])
    
    rpy_np = np.array(rpy_list)
    
    # Calculate relative time from first timestamp
    relative_time = timestamps_np - timestamps_np[0]
    
    # Create plots
    create_plots(positions_np, rpy_np, relative_time)
    
    # Save data to file
    save_data(positions_np, orientations_np, rpy_np, timestamps_np)
    
    sys.exit(0)

def create_plots(positions, rpy, time):
    """Create position/orientation vs time plots and 3D trajectory"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Position vs Time
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(time, positions[:, 0], 'r-', label='X')
    plt.plot(time, positions[:, 1], 'g-', label='Y')
    plt.plot(time, positions[:, 2], 'b-', label='Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Position vs Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Roll, Pitch, Yaw vs Time
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(time, np.degrees(rpy[:, 0]), 'r-', label='Roll')
    plt.plot(time, np.degrees(rpy[:, 1]), 'g-', label='Pitch')
    plt.plot(time, np.degrees(rpy[:, 2]), 'b-', label='Yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.title('Orientation vs Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: X vs Time
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(time, positions[:, 0], 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title('X Position vs Time')
    plt.grid(True)
    
    # Plot 4: Y vs Time
    ax4 = plt.subplot(2, 3, 4)
    plt.plot(time, positions[:, 1], 'g-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position vs Time')
    plt.grid(True)
    
    # Plot 5: Z vs Time
    ax5 = plt.subplot(2, 3, 5)
    plt.plot(time, positions[:, 2], 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.title('Z Position vs Time')
    plt.grid(True)
    
    # Plot 6: 3D Trajectory
    ax6 = plt.subplot(2, 3, 6, projection='3d')
    
    # Create color map based on time for trajectory
    colors = time / time[-1] if len(time) > 1 else [0]
    
    scatter = ax6.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                         c=colors, cmap='viridis', s=20, alpha=0.6)
    
    # Plot trajectory line
    line = ax6.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   'b-', alpha=0.4, linewidth=1)
    
    # Mark start and end points
    ax6.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='green', s=100, marker='o', label='Start')
    ax6.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               c='red', s=100, marker='x', label='End')
    
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_zlabel('Z (m)')
    ax6.set_title('3D Trajectory')
    
    # Set equal aspect ratio (no axis scaling)
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(), 
                         positions[:, 1].max()-positions[:, 1].min(), 
                         positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
    
    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
    
    ax6.set_xlim(mid_x - max_range, mid_x + max_range)
    ax6.set_ylim(mid_y - max_range, mid_y + max_range)
    ax6.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax6.legend()
    plt.colorbar(scatter, ax=ax6, label='Normalized Time')
    
    plt.tight_layout()
    plt.savefig('trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_data(positions, orientations, rpy, timestamps):
    """Save all data to numpy files"""
    # Save positions and orientations
    np.save('positions.npy', positions)
    np.save('orientations.npy', orientations)
    np.save('rpy_angles.npy', rpy)
    np.save('timestamps.npy', timestamps)
    
    # Also save transform matrices for backward compatibility
    transform_matrices = []
    for i in range(len(positions)):
        transform_matrix = quaternion_matrix(orientations[i])
        transform_matrix[0, 3] = positions[i, 0]
        transform_matrix[1, 3] = positions[i, 1]
        transform_matrix[2, 3] = positions[i, 2]
        transform_matrices.append(transform_matrix)
    
    np.save('odom_transform_matrices.npy', np.array(transform_matrices))
    
    rospy.loginfo(f"Successfully saved data for {len(positions)} poses")
    rospy.loginfo("Files created: positions.npy, orientations.npy, rpy_angles.npy, timestamps.npy, odom_transform_matrices.npy")
    rospy.loginfo("Plot saved as: trajectory_analysis.png")

def odom_callback(msg: Odometry):
    """Process odom topic, extract pose and store data"""
    # Extract position
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z
    
    # Extract quaternion
    quat = [
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    ]
    
    # Store data
    positions.append([x, y, z])
    orientations.append(quat)
    timestamps.append(msg.header.stamp.to_sec())
    
    # Print status every 100 messages
    if len(positions) % 100 == 0:
        rospy.loginfo(f"Collected {len(positions)} pose samples")

def main():
    # Initialize node
    rospy.init_node('odom_visualization', anonymous=True)
    
    # Register exit signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Subscribe to odom topic (fixed topic name)
    # odom_topic = '/lio_sam/mapping/odometry' #lio_sam
    odom_topic = '/Odometry' #FASTLIO
    rospy.Subscriber( odom_topic, Odometry, odom_callback)
    
    rospy.loginfo(f"Started listening to {odom_topic} topic... (Press Ctrl+C to save and exit)")
    rospy.loginfo("Will create:")
    rospy.loginfo("  - Position and orientation vs time plots")
    rospy.loginfo("  - 3D trajectory plot with equal axis scaling")
    rospy.loginfo("  - Data files: positions.npy, orientations.npy, rpy_angles.npy, timestamps.npy")
    
    # Keep node running
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass