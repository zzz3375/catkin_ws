import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
plt.rcParams['font.sans-serif'] = ['Times New Roman']

def read_poses_from_pcd(pcd_file):
    """
    Read poses from PCD file
    Assumes each point in PCD represents a pose (x, y, z, [optional orientation])
    Returns: numpy array of positions (N, 3)
    """
    try:
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points)
        return points
    except Exception as e:
        print(f"Error reading PCD file: {e}")
        return None

def calculate_end_to_end_error(poses, end_idx):
    """
    Calculate end-to-end translation error between start and selected end pose
    """
    if poses is None or len(poses) < 2 or end_idx >= len(poses) or end_idx < 0:
        print("Error: Invalid poses or end index")
        return None
    
    start_pose = poses[0]  # First pose (x, y, z)
    end_pose = poses[end_idx]   # Selected end pose
    
    # Calculate Euclidean distance (translation error)
    translation_error = np.linalg.norm(end_pose - start_pose)
    
    return translation_error, start_pose, end_pose

def plot_trajectories(poses):
    if poses is None or len(poses) < 2:
        print("Error: Not enough poses for visualization")
        return

    # Create figure with 1 row and 4 columns (3D + X + Y + Z)
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle('Trajectory Visualization', fontsize=16)
    
    # 3D Trajectory Subplot
    ax_3d = fig.add_subplot(141, projection='3d')
    
    # Plot full trajectory
    ax_3d.plot(poses[:, 0], poses[:, 1], poses[:, 2], 'gray', linewidth=1, label='Trajectory')
    
    # Mark start point
    ax_3d.scatter(poses[0, 0], poses[0, 1], poses[0, 2], c='green', s=100, marker='*', 
               label='Start Point')
    
    # Initialize end point marker (default to last point)
    selected_idx = len(poses) - 1
    end_marker_3d = ax_3d.scatter(poses[selected_idx, 0], poses[selected_idx, 1], 
                           poses[selected_idx, 2], c='red', s=100, marker='o',
                           label='End Point')
    
    # Set 3D axis labels
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.legend()
    
    # Keep equal scale for 3D axes
    x_range = np.ptp(poses[:, 0])
    y_range = np.ptp(poses[:, 1])
    z_range = np.ptp(poses[:, 2])
    max_range = max(x_range, y_range, z_range)
    
    mid_x = (poses[:, 0].min() + poses[:, 0].max()) / 2
    mid_y = (poses[:, 1].min() + poses[:, 1].max()) / 2
    mid_z = (poses[:, 2].min() + poses[:, 2].max()) / 2
    
    ax_3d.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax_3d.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax_3d.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    # Sequence indices (x-axis for individual plots)
    indices = np.arange(len(poses))
    marker_size = 50
    
    # X-axis trajectory subplot
    ax_x = fig.add_subplot(142)
    ax_x.plot(indices, poses[:, 0], 'r-', linewidth=1, label='X Position')
    ax_x.scatter(indices[0], poses[0, 0], c='green', s=marker_size, marker='*', label='Start')
    marker_x = ax_x.scatter(indices[selected_idx], poses[selected_idx, 0], c='red', s=marker_size, marker='o')
    ax_x.set_xlabel('Sequence Number')
    ax_x.set_ylabel('X (m)')
    ax_x.grid(True)
    ax_x.legend()
    
    # Y-axis trajectory subplot
    ax_y = fig.add_subplot(143)
    ax_y.plot(indices, poses[:, 1], 'g-', linewidth=1, label='Y Position')
    ax_y.scatter(indices[0], poses[0, 1], c='green', s=marker_size, marker='*')
    marker_y = ax_y.scatter(indices[selected_idx], poses[selected_idx, 1], c='red', s=marker_size, marker='o')
    ax_y.set_xlabel('Sequence Number')
    ax_y.set_ylabel('Y (m)')
    ax_y.grid(True)
    ax_y.legend()
    
    # Z-axis trajectory subplot
    ax_z = fig.add_subplot(144)
    ax_z.plot(indices, poses[:, 2], 'b-', linewidth=1, label='Z Position')
    ax_z.scatter(indices[0], poses[0, 2], c='green', s=marker_size, marker='*')
    marker_z = ax_z.scatter(indices[selected_idx], poses[selected_idx, 2], c='red', s=marker_size, marker='o')
    ax_z.set_xlabel('Sequence Number')
    ax_z.set_ylabel('Z (m)')
    ax_z.grid(True)
    ax_z.legend()

    # Error display text
    error_text = fig.text(0.5, 0.01, '', ha='center', fontsize=10)

    def update_end_point(idx):
        """Update end point markers across all plots and recalculate error"""
        nonlocal selected_idx, end_marker_3d, marker_x, marker_y, marker_z
        selected_idx = idx
        
        # Update 3D marker
        end_marker_3d.remove()
        end_marker_3d = ax_3d.scatter(poses[idx, 0], poses[idx, 1], poses[idx, 2],
                               c='red', s=100, marker='o', label='End Point')
        
        # Update X marker
        marker_x.remove()
        marker_x = ax_x.scatter(indices[idx], poses[idx, 0], c='red', s=marker_size, marker='o')
        
        # Update Y marker
        marker_y.remove()
        marker_y = ax_y.scatter(indices[idx], poses[idx, 1], c='red', s=marker_size, marker='o')
        
        # Update Z marker
        marker_z.remove()
        marker_z = ax_z.scatter(indices[idx], poses[idx, 2], c='red', s=marker_size, marker='o')
        
        # Recalculate and display error
        result = calculate_end_to_end_error(poses, idx)
        if result:
            error, start, end = result
            error_text.set_text(
                f"End-to-end error: {error:.4g}m | Start: [{start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f}] | "
                f"End: [{end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f}] (Index: {idx})"
            )
        fig.canvas.draw_idle()

    def on_click(event):
        """Handle mouse click to select closest point in any subplot"""
        # if event.inaxes == ax_3d:
        #     # Handle 3D plot click
        #     x, y = event.xdata, event.ydata
        #     ray_origin, ray_direction = ax_3d.unproject_ray(x, y)
            
        #     closest_dist = float('inf')
        #     closest_idx = selected_idx
        #     for i, (px, py, pz) in enumerate(poses):
        #         v = np.array([px, py, pz]) - ray_origin
        #         dist = np.linalg.norm(np.cross(v, ray_direction))
        #         if dist < closest_dist:
        #             closest_dist = dist
        #             closest_idx = i
        #     update_end_point(closest_idx)
            
        if event.inaxes in [ax_x, ax_y, ax_z]:
            # Handle 2D plot click (use x coordinate as sequence reference)
            click_idx = event.xdata
            closest_idx = np.argmin(np.abs(indices - click_idx))
            update_end_point(closest_idx)

    # Add buttons for quick selection
    ax_last = plt.axes([0.8, 0.05, 0.15, 0.03])
    btn_last = Button(ax_last, 'Last Point')
    btn_last.on_clicked(lambda event: update_end_point(len(poses) - 1))

    ax_first = plt.axes([0.05, 0.05, 0.15, 0.03])
    btn_first = Button(ax_first, 'First Point')
    btn_first.on_clicked(lambda event: update_end_point(0))

    # Bind click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Initial error calculation
    update_end_point(selected_idx)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def main():
    # Replace with your PCD file path
    # pcd_file = "/home/zzz2004/Downloads/LOAM/trajectory.pcd"
    pcd_file = "/home/zzz2004/PIN_SLAM/experiments/test_pin_2025-10-24_21-29-30/odom_poses.ply"
    
    # Read poses from PCD
    poses = read_poses_from_pcd(pcd_file)
    
    if poses is not None:
        print(f"Loaded {len(poses)} poses from {pcd_file}")
        
        # Plot all trajectories
        plot_trajectories(poses)
        
        # Calculate default end-to-end error (start to last point)
        result = calculate_end_to_end_error(poses, len(poses) - 1)
        if result is not None:
            error, start_pose, end_pose = result
            
            print("\n" + "="*50)
            print("END-TO-END TRANSLATION ERROR RESULTS")
            print("="*50)
            print(f"Start pose: [{start_pose[0]:.4g}, {start_pose[1]:.4g}, {start_pose[2]:.4g}]")
            print(f"End pose:   [{end_pose[0]:.4g}, {end_pose[1]:.4g}, {end_pose[2]:.4g}]")
            print("-"*50)
            print(f"End-to-end translation error: {error:.4g} meters")
            print("="*50)

if __name__ == "__main__":
    main()