#!/usr/bin/env python3

import rosbag
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from collections import deque
from scipy.spatial.transform import Rotation as R
import threading

# Constants
G_MS2 = 9.81  # Gravity constant

def exp_map(omega):
    """Exponential map for SO(3) - converts angular velocity to rotation matrix"""
    angle = np.linalg.norm(omega)
    if angle < 1e-10:
        return np.eye(3)
    
    axis = omega / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R_mat = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R_mat

class IMUPreintegrator:
    def __init__(self):
        self.imu_buffer = deque()
        self.imu_initialized = False
        self.imu_lock = threading.Lock()
        
    def add_imu_message(self, imu_msg):
        """Add IMU message to buffer"""
        with self.imu_lock:
            self.imu_buffer.append(imu_msg)
            if not self.imu_initialized and len(self.imu_buffer) >= 2:
                self.imu_initialized = True
    
    def preintegrate_imu(self, last_pose: np.ndarray):
        """
        使用中值积分法预积分IMU测量值
        
        Args:
            last_pose: 上一时刻的位姿 (4x4 矩阵)
        Returns: 
            预测的位姿变换 (4x4 矩阵)，表示相对运动
        """
        g = np.array([0, 0, 0]) * G_MS2
        R0 = last_pose[:3, :3]

        if not self.imu_initialized:
            return np.eye(4)  # 如果没有IMU数据，返回单位矩阵
        
        if not self.imu_buffer:
            return np.eye(4)
        
        # 初始化预积分量（在机体坐标系下）
        delta_p = np.zeros(3)  # 相对位置变化
        delta_v = np.zeros(3)  # 相对速度变化  
        delta_R = np.eye(3)    # 相对旋转

        gyro_bias = np.array([0.010554, -0.010818 , 0.018882])
        # gyro_bias = np.array([0.010554, -0.010818 , 0.016])
        accel_bias = np.array([0.081176, -0.0065639 , 0.98246])

        with self.imu_lock:
            if len(self.imu_buffer) < 2: 
                return np.eye(4)
                
            imu_k = None
            
            for i, imu_msg in enumerate(self.imu_buffer):
                if i == 0: 
                    imu_k = imu_msg
                    continue           

                imu_k1 = imu_msg
                
                # 时间间隔
                t_k = imu_k.header.stamp.to_sec()
                t_k1 = imu_k1.header.stamp.to_sec()
                dt = t_k1 - t_k
                
                if dt <= 0:
                    continue
                
                # 提取角速度和加速度测量值
                gyro_k = np.array([
                    imu_k.angular_velocity.x - gyro_bias[0],
                    imu_k.angular_velocity.y - gyro_bias[1], 
                    imu_k.angular_velocity.z - gyro_bias[2]
                ])
                
                gyro_k1 = np.array([
                    imu_k1.angular_velocity.x - gyro_bias[0],
                    imu_k1.angular_velocity.y - gyro_bias[1],
                    imu_k1.angular_velocity.z - gyro_bias[2]
                ])
                
                # Apply gravity scaling (since you observed z ≈ +1)
                accel_k = np.array([
                    imu_k.linear_acceleration.x - accel_bias[0],
                    imu_k.linear_acceleration.y - accel_bias[1],
                    imu_k.linear_acceleration.z - accel_bias[2]
                ]) * G_MS2  # Convert from g to m/s²
                
                accel_k1 = np.array([
                    imu_k1.linear_acceleration.x - accel_bias[0],
                    imu_k1.linear_acceleration.y - accel_bias[1],
                    imu_k1.linear_acceleration.z - accel_bias[2]
                ]) * G_MS2  # Convert from g to m/s²
                
                # 中值积分
                gyro_mid = 0.5 * (gyro_k + gyro_k1)
                accel_mid = 0.5 * (accel_k + accel_k1)
                
                # 1. 更新旋转
                delta_theta = gyro_mid * dt
                delta_R_k1 = delta_R @ exp_map(delta_theta)
                
                # 2. 更新速度
                # 注意：加速度需要转换到世界坐标系（使用平均旋转）
                R_mid = delta_R @ exp_map(0.5 * delta_theta)
                
                delta_v_k1 = delta_v + (R_mid @ accel_mid ) * dt
                
                # 3. 更新位置
                delta_p_k1 = delta_p + delta_v * dt + 0.5 * (R_mid @ accel_mid ) * dt * dt
                
                # 为下一次迭代更新状态
                delta_R = delta_R_k1
                delta_v = delta_v_k1
                delta_p = delta_p_k1
                
                imu_k = imu_k1
            
            # 创建变换矩阵
            transform = np.eye(4)
            transform[0:3, 0:3] = delta_R  # 旋转部分
            transform[0:3, 3] = delta_p    # 平移部分

            self.imu_buffer.clear()
            
            return last_pose @ transform

def extract_euler_angles(rotation_matrix):
    """Extract roll, pitch, yaw from rotation matrix"""
    # Using scipy for more robust conversion
    r = R.from_matrix(rotation_matrix)
    return r.as_euler('xyz', degrees=True)  # roll, pitch, yaw in degrees

def visualize_imu_trajectory_3d(bag_file, imu_topic='/imu/data', max_messages=10000):
    """
    Visualize IMU-integrated trajectory in 3D using matplotlib
    
    Args:
        bag_file: Path to ROS bag file
        imu_topic: IMU topic name
        max_messages: Maximum number of IMU messages to process
    """
    
    print(f"Reading IMU data from: {bag_file}")
    
    # Read IMU data
    imu_messages = []
    timestamps = []
    
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            topic_info = bag.get_type_and_topic_info()
            if imu_topic not in topic_info[1]:
                print(f"Topic '{imu_topic}' not found!")
                print("Available topics:")
                for topic, info in topic_info[1].items():
                    print(f"  {topic} - {info.message_count} messages")
                return
            
            total_messages = min(topic_info[1][imu_topic].message_count, max_messages)
            
            with tqdm(total=total_messages, desc="Reading IMU data", unit="msg") as pbar:
                message_count = 0
                for topic, msg, t in bag.read_messages(topics=[imu_topic]):
                    if message_count >= max_messages:
                        break
                    imu_messages.append(msg)
                    timestamps.append(t.to_sec())
                    message_count += 1
                    pbar.update(1)
                    
    except Exception as e:
        print(f"Error reading bag file: {e}")
        return
    
    if not imu_messages:
        print("No IMU data found!")
        return
    
    print(f"Processing {len(imu_messages)} IMU measurements...")
    
    # Initialize preintegrator and trajectory
    preintegrator = IMUPreintegrator()
    trajectory_poses = []
    euler_angles = []  # Store roll, pitch, yaw
    
    # Start with identity pose
    current_pose = np.eye(4)
    trajectory_poses.append(current_pose.copy())
    
    # Process IMU data in chunks
    chunk_size = 100
    
    with tqdm(total=len(imu_messages), desc="Integrating IMU", unit="msg") as pbar:
        for i in range(0, len(imu_messages), chunk_size):
            end_idx = min(i + chunk_size, len(imu_messages))
            chunk_data = imu_messages[i:end_idx]
            
            # Add messages to preintegrator
            for imu_msg in chunk_data:
                preintegrator.add_imu_message(imu_msg)
            
            # Preintegrate to get new pose
            current_pose = preintegrator.preintegrate_imu(current_pose)
            trajectory_poses.append(current_pose.copy())
            
            # Extract Euler angles
            euler = extract_euler_angles(current_pose[:3, :3])
            euler[euler>160] = euler[euler>160] - 360
            euler_angles.append(euler)
            
            pbar.update(len(chunk_data))
    
    print(f"Generated {len(trajectory_poses)} poses")
    
    # Extract trajectory data
    positions = np.array([pose[:3, 3] for pose in trajectory_poses])
    euler_angles = np.array(euler_angles)
    
    # Create 3D visualization
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 3D Trajectory Plot
    ax1 = fig.add_subplot(231, projection='3d')
    
    # Color trajectory by time
    colors = np.arange(len(positions))
    
    scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                         c=colors, cmap='viridis', s=20, alpha=0.8)
    
    # Plot trajectory line
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
             'b-', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D IMU Trajectory')
    plt.colorbar(scatter, ax=ax1, label='Time Step')
    
    # Set equal aspect ratio
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(), 
                         positions[:, 1].max()-positions[:, 1].min(), 
                         positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
    
    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 2. 2D Projections
    # XY projection
    ax2 = fig.add_subplot(234)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7)
    ax2.scatter(positions[:, 0], positions[:, 1], c=colors, cmap='viridis', s=10)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Projection')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # XZ projection
    ax3 = fig.add_subplot(235)
    ax3.plot(positions[:, 0], positions[:, 2], 'r-', alpha=0.7)
    ax3.scatter(positions[:, 0], positions[:, 2], c=colors, cmap='viridis', s=10)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Projection')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # YZ projection
    ax4 = fig.add_subplot(236)
    ax4.plot(positions[:, 1], positions[:, 2], 'g-', alpha=0.7)
    ax4.scatter(positions[:, 1], positions[:, 2], c=colors, cmap='viridis', s=10)
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('YZ Projection')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    # 3. Euler Angles Plot
    ax5 = fig.add_subplot(232)
    time_steps = np.arange(len(euler_angles))
    ax5.plot(time_steps, euler_angles[:, 0], 'r-', label='Roll', alpha=0.8)
    ax5.plot(time_steps, euler_angles[:, 1], 'g-', label='Pitch', alpha=0.8)
    ax5.plot(time_steps, euler_angles[:, 2], 'b-', label='Yaw', alpha=0.8)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Angle (degrees)')
    ax5.set_title('Euler Angles')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 4. Position Components
    ax6 = fig.add_subplot(233)
    time_steps_pos = np.arange(len(positions))
    ax6.plot(time_steps_pos, positions[:, 0], 'r-', label='X', alpha=0.8)
    ax6.plot(time_steps_pos, positions[:, 1], 'g-', label='Y', alpha=0.8)
    ax6.plot(time_steps_pos, positions[:, 2], 'b-', label='Z', alpha=0.8)
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Position (m)')
    ax6.set_title('Position Components')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n=== Trajectory Statistics ===")
    print(f"Total distance traveled: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.2f} m")
    print(f"Final position: {positions[-1]}")
    print(f"Position range - X: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f} m")
    print(f"Position range - Y: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f} m")
    print(f"Position range - Z: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f} m")
    print(f"Euler angles range - Roll: {euler_angles[:, 0].min():.1f} to {euler_angles[:, 0].max():.1f} deg")
    print(f"Euler angles range - Pitch: {euler_angles[:, 1].min():.1f} to {euler_angles[:, 1].max():.1f} deg")
    print(f"Euler angles range - Yaw: {euler_angles[:, 2].min():.1f} to {euler_angles[:, 2].max():.1f} deg")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize IMU-integrated trajectory in 3D')
    # parser.add_argument('bag_file', help='Path to the ROS bag file')
    parser.add_argument('--topic', default='/livox/imu', help='IMU topic name')
    parser.add_argument('--max_messages', type=int, default=70000, 
                       help='Maximum number of IMU messages to process')
    
    args = parser.parse_args()
    
    # if not os.path.exists(args.bag_file):
    #     print(f"Error: Bag file '{args.bag_file}' not found!")
    #     exit(1)
    
    visualize_imu_trajectory_3d("/home/zzz2004/catkin_ws/chuangye_0910_high_01.bag", args.topic, args.max_messages)