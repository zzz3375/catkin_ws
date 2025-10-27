import open3d as o3d
import numpy as np

def show_coordinate_frame():
    # 创建一个坐标系
    # size 参数表示坐标轴的长度
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )
    
    # 可以添加一个球体作为原点标记，让原点更明显
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色球体
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D 坐标系示例")
    
    # 添加坐标系和球体到可视化窗口
    vis.add_geometry(coordinate_frame)
    vis.add_geometry(sphere)
    
    # 设置视角（可选）
    ctr = vis.get_view_control()
    ctr.set_front([1, 1, 1])  # 相机位置
    ctr.set_lookat([0, 0, 0])  # 看向原点
    ctr.set_up([0, 1, 0])      # 相机上方向
    ctr.set_zoom(0.8)          # 缩放
    
    # 运行可视化
    print("可视化窗口已打开，按 'q' 键退出")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 检查Open3D版本
    print(f"Open3D 版本: {o3d.__version__}")
    show_coordinate_frame()
