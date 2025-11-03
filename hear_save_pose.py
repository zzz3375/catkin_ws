#!/usr/bin/env python3
import rospy
import signal
import sys
import numpy as np
import open3d as o3d
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# 存储所有位姿数据的列表 [x, y, z, roll, pitch, yaw]
poses = []

def signal_handler(sig, frame):
    """处理退出信号，保存包含自定义属性的PCD文件"""
    if not poses:
        rospy.logwarn("没有收集到任何位姿数据，不保存文件")
        sys.exit(0)
    
    # 转换为numpy数组 (n, 6) 形状: [x, y, z, roll, pitch, yaw]
    poses_np = np.array(poses, dtype=np.float64)
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 设置位置信息 (x, y, z)
    pcd.points = o3d.utility.Vector3dVector(poses_np[:, :3])
    
    # 添加自定义属性：roll, pitch, yaw（以numpy数组形式存储为点云的用户数据）
    # Open3D中使用float64类型存储自定义属性，确保精度
    pcd.user_data = {
        "roll": poses_np[:, 3],
        "pitch": poses_np[:, 4],
        "yaw": poses_np[:, 5]
    }
    
    # 保存为非压缩PCD格式（ASCII格式，确保属性可读取）
    filename = "odom_poses_with_rpy.pcd"
    # 强制使用ASCII格式保存，避免压缩导致自定义属性丢失
    o3d.io.write_point_cloud(
        filename, 
        pcd, 
        write_ascii=True,  # 非压缩ASCII格式
        compressed=False   # 显式关闭压缩
    )
    
    rospy.loginfo(
        f"成功保存 {len(poses)} 个位姿到 {filename}\n"
        f"包含信息: x, y, z (位置) | roll, pitch, yaw (欧拉角，弧度)"
    )
    sys.exit(0)

def odom_callback(msg: Odometry):
    """处理odom话题，提取位置和姿态（转换为欧拉角）"""
    # 提取位置信息 (x, y, z)
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z
    
    # 提取四元数并转换为roll, pitch, yaw（弧度）
    quat = [
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    ]
    roll, pitch, yaw = euler_from_quaternion(quat)  # 顺序: roll (x), pitch (y), yaw (z)
    
    # 存储位置和欧拉角
    poses.append([x, y, z, roll, pitch, yaw])
    
    # 每100个点打印一次状态
    if len(poses) % 100 == 0:
        rospy.loginfo(f"已收集 {len(poses)} 个位姿点（含roll/pitch/yaw）")

def main():
    # 初始化节点
    rospy.init_node('odom_to_pcd_with_rpy', anonymous=True)
    
    # 注册退出信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 订阅odom话题
    rospy.Subscriber('/odom', Odometry, odom_callback)
    
    rospy.loginfo("开始监听 /odom 话题... (按 Ctrl+C 保存并退出)\n"
                  "将保存: x, y, z (位置) 和 roll, pitch, yaw (欧拉角，弧度)")
    
    # 保持节点运行
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass