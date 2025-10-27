#!/usr/bin/env python3

import rosbag
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def simple_imu_plot_with_progress(bag_file, imu_topic='/imu/data'):
    """
    Simple IMU acceleration and angular velocity plotter with progress bars
    """
    timestamps = []
    accel_x, accel_y, accel_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    
    print(f"Processing {bag_file}...")
    
    with rosbag.Bag(bag_file, 'r') as bag:
        # Get message count for progress bar
        topic_info = bag.get_type_and_topic_info()[1]
        if imu_topic not in topic_info:
            print(f"Error: Topic '{imu_topic}' not found!")
            print("Available topics:")
            for topic in topic_info.keys():
                print(f"  {topic}")
            return
        
        total_messages = topic_info[imu_topic].message_count
        start_time = None
        msg: Imu
        # Read messages with progress bar
        with tqdm(total=total_messages, unit='msg', desc='Reading IMU data') as pbar:
            for topic, msg, t in bag.read_messages(topics=[imu_topic]):
                
                if start_time is None:
                    start_time = t.to_sec()
                
                current_time = t.to_sec() - start_time
                accel = msg.linear_acceleration
                gyro = msg.angular_velocity
                
                timestamps.append(current_time)
                accel_x.append(accel.x)
                accel_y.append(accel.y)
                accel_z.append(accel.z)
                gyro_x.append(gyro.x)
                gyro_y.append(gyro.y)
                gyro_z.append(gyro.z)
                
                pbar.update(1)
    
    # Convert to numpy arrays for statistics
    accel_x_arr = np.array(accel_x)[:3000]
    accel_y_arr = np.array(accel_y)[:3000]
    accel_z_arr = np.array(accel_z)[:3000]
    gyro_x_arr = np.array(gyro_x)[:3000]
    gyro_y_arr = np.array(gyro_y)[:3000]
    gyro_z_arr = np.array(gyro_z)[:3000]
    
    # Print statistics
    print("\n=== Acceleration Statistics (m/s²) ===")
    print(f"X-axis: mean={np.mean(accel_x_arr):.5g}, std={np.std(accel_x_arr):.5g}, min={np.min(accel_x_arr):.5g}, max={np.max(accel_x_arr):.5g}")
    print(f"Y-axis: mean={np.mean(accel_y_arr):.5g}, std={np.std(accel_y_arr):.5g}, min={np.min(accel_y_arr):.5g}, max={np.max(accel_y_arr):.5g}")
    print(f"Z-axis: mean={np.mean(accel_z_arr):.5g}, std={np.std(accel_z_arr):.5g}, min={np.min(accel_z_arr):.5g}, max={np.max(accel_z_arr):.5g}")
    
    print("\n=== Gyroscope Statistics (rad/s) ===")
    print(f"X-axis: mean={np.mean(gyro_x_arr):.5g}, std={np.std(gyro_x_arr):.5g}, min={np.min(gyro_x_arr):.5g}, max={np.max(gyro_x_arr):.5g}")
    print(f"Y-axis: mean={np.mean(gyro_y_arr):.5g}, std={np.std(gyro_y_arr):.5g}, min={np.min(gyro_y_arr):.5g}, max={np.max(gyro_y_arr):.5g}")
    print(f"Z-axis: mean={np.mean(gyro_z_arr):.5g}, std={np.std(gyro_z_arr):.5g}, min={np.min(gyro_z_arr):.5g}, max={np.max(gyro_z_arr):.5g}")
    
    # Create comprehensive plot
    print("Generating plots...")
    fig = plt.figure(figsize=(16, 12))
    
    # Acceleration plots
    plt.subplot(3, 2, 1)
    plt.plot(timestamps, accel_x, 'r-', label='X-axis', linewidth=1, alpha=0.8)
    plt.plot(timestamps, accel_y, 'g-', label='Y-axis', linewidth=1, alpha=0.8)
    plt.plot(timestamps, accel_z, 'b-', label='Z-axis', linewidth=1, alpha=0.8)
    plt.ylabel('Acceleration (m/s²)')
    plt.title('IMU Linear Acceleration - All Axes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gyroscope plots
    plt.subplot(3, 2, 2)
    plt.plot(timestamps, gyro_x, 'r-', label='X-axis', linewidth=1, alpha=0.8)
    plt.plot(timestamps, gyro_y, 'g-', label='Y-axis', linewidth=1, alpha=0.8)
    plt.plot(timestamps, gyro_z, 'b-', label='Z-axis', linewidth=1, alpha=0.8)
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('IMU Angular Velocity - All Axes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Individual acceleration plots
    colors = ['red', 'green', 'blue']
    accel_data = [accel_x, accel_y, accel_z]
    gyro_data = [gyro_x, gyro_y, gyro_z]
    titles = ['X-axis', 'Y-axis', 'Z-axis']
    
    for i, (color, accel, gyro, title) in enumerate(zip(colors, accel_data, gyro_data, titles)):
        # Acceleration subplots
        plt.subplot(3, 3, 4 + i)
        plt.plot(timestamps, accel, color=color, linewidth=1, alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel(f'Acceleration {title} (m/s²)')
        plt.title(f'Acceleration {title}')
        plt.grid(True, alpha=0.3)
        
        # Gyroscope subplots
        plt.subplot(3, 3, 7 + i)
        plt.plot(timestamps, gyro, color=color, linewidth=1, alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel(f'Angular Velocity {title} (rad/s)')
        plt.title(f'Gyroscope {title}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create separate detailed plots for better analysis
    create_detailed_plots(timestamps, accel_data, gyro_data, titles)

def create_detailed_plots(timestamps, accel_data, gyro_data, titles):
    """Create more detailed individual plots for analysis"""
    
    # Detailed acceleration plot
    plt.figure(figsize=(14, 10))
    
    for i, (accel, title, color) in enumerate(zip(accel_data, titles, ['red', 'green', 'blue'])):
        plt.subplot(2, 3, i + 1)
        plt.plot(timestamps, accel, color=color, linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel(f'Acceleration (m/s²)')
        plt.title(f'Linear Acceleration - {title}')
        plt.grid(True, alpha=0.3)
        
        # Add statistics to plot
        accel_arr = np.array(accel)
        stats_text = f'Mean: {np.mean(accel_arr):.5g}\nStd: {np.std(accel_arr):.5g}\nMax: {np.max(accel_arr):.5g}\nMin: {np.min(accel_arr):.5g}'
        plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Detailed gyroscope plot
    for i, (gyro, title, color) in enumerate(zip(gyro_data, titles, ['red', 'green', 'blue'])):
        plt.subplot(2, 3, i + 4)
        plt.plot(timestamps, gyro, color=color, linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel(f'Angular Velocity (rad/s)')
        plt.title(f'Angular Velocity - {title}')
        plt.grid(True, alpha=0.3)
        
        # Add statistics to plot
        gyro_arr = np.array(gyro)
        stats_text = f'Mean: {np.mean(gyro_arr):.5g}\nStd: {np.std(gyro_arr):.5g}\nMax: {np.max(gyro_arr):.5g}\nMin: {np.min(gyro_arr):.5g}'
        plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Combined time-frequency analysis
    create_combined_analysis(timestamps, accel_data, gyro_data, titles)

def create_combined_analysis(timestamps, accel_data, gyro_data, titles):
    """Create combined analysis plots"""
    fig = plt.figure(figsize=(15, 8))
    
    # Time domain comparison
    plt.subplot(2, 1, 1)
    for i, (accel, title, color) in enumerate(zip(accel_data, titles, ['red', 'green', 'blue'])):
        plt.plot(timestamps, accel, color=color, linewidth=1, alpha=0.7, label=f'Accel {title}')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Time Domain - Acceleration vs Gyroscope')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    for i, (gyro, title, color) in enumerate(zip(gyro_data, titles, ['red', 'green', 'blue'])):
        plt.plot(timestamps, gyro, color=color, linewidth=1, alpha=0.7, label=f'Gyro {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('bag_file', help='Path to ROS bag file')
    parser.add_argument('--topic', default='/livox/imu', help='IMU topic name')
    
    args = parser.parse_args()
    simple_imu_plot_with_progress("/home/zzz2004/catkin_ws/chuangye_0910_high_01.bag", args.topic)