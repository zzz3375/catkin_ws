"""
open3d python icp
source_mesh = /home/zzz2004/catkin_ws/BIM.obj
target_pcd = ~/Downloads/LOAM/GlobalMap.pcd

mesh = o3d.io.read_triangle_model(source_mesh)
mesh = something api about read_triangle_model(mesh)
source_pcd = mesh.vertices
souce_pcd's normal = mesh vertice normal

target_pcd = read pcd
compute target normal

point to plane icp
print icp results 

save output to ./


"""

import open3d as o3d
import numpy as np
import os

def main():
    # File paths
    source_mesh_path = "/home/zzz2004/catkin_ws/BIM.obj"
    target_pcd_path = os.path.expanduser("/home/zzz2004/Downloads/LOAM/GlobalMap.pcd")
    output_dir = "./"
    
    # Read source mesh
    print("Loading source mesh...")
    mesh = o3d.io.read_triangle_mesh(source_mesh_path)
    
    if not mesh.has_vertices():
        print("Error: Failed to load mesh or mesh has no vertices")
        return
    
    # Convert mesh to point cloud with normals
    print("Converting mesh to point cloud with normals...")
    mesh.compute_vertex_normals()
    
    # Create point cloud from mesh vertices
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = mesh.vertices
    source_pcd.normals = mesh.vertex_normals
    
    # Downsample source point cloud for faster processing (optional)
    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.05)
    
    # Read target point cloud
    print("Loading target point cloud...")
    target_pcd = o3d.io.read_point_cloud(target_pcd_path)
    
    if target_pcd.is_empty():
        print("Error: Failed to load target point cloud")
        return
    
    # Compute normals for target point cloud
    print("Computing target point cloud normals...")
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Downsample target point cloud (optional)
    target_pcd = target_pcd.voxel_down_sample(voxel_size=0.05)
    
    # Make sure both point clouds are aligned (you might need to adjust this)
    print("Preprocessing point clouds...")
    
    # Optional: Remove outliers
    source_pcd, _ = source_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    target_pcd, _ = target_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Point-to-plane ICP registration
    print("Running point-to-plane ICP...")
    
    # Set ICP parameters
    threshold = 0.1  # Maximum correspondence distance
    trans_init = np.identity(4)  # Initial transformation (identity)
    
    # Perform point-to-plane ICP
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    # Print ICP results
    print("\n=== ICP Results ===")
    print(f"Fitness: {reg_p2l.fitness:.6f}")
    print(f"Inlier RMSE: {reg_p2l.inlier_rmse:.6f}")
    # print(f"Number of iterations: {reg_p2l.num_iteration}")
    
    print("\nTransformation matrix:")
    print(reg_p2l.transformation)
    
    # Apply transformation to source point cloud
    source_pcd_transformed = source_pcd.transform(reg_p2l.transformation)
    
    # Save results
    print(f"\nSaving results to {output_dir}")
    
    # Save transformed source point cloud
    o3d.io.write_point_cloud(os.path.join(output_dir, "source_transformed.pcd"), source_pcd_transformed)
    
    # Save transformation matrix
    np.savetxt(os.path.join(output_dir, "icp_transformation.txt"), reg_p2l.transformation)
    
    # Save ICP results summary
    with open(os.path.join(output_dir, "icp_results.txt"), "w") as f:
        f.write("ICP Registration Results\n")
        f.write("========================\n")
        f.write(f"Fitness: {reg_p2l.fitness:.6f}\n")
        f.write(f"Inlier RMSE: {reg_p2l.inlier_rmse:.6f}\n")
        # f.write(f"Number of iterations: {reg_p2l.num_iteration}\n")
        f.write("\nTransformation matrix:\n")
        f.write(str(reg_p2l.transformation))
    
    # Visualize results (optional)
    print("\nVisualizing results...")
    
    # Color point clouds for visualization
    source_pcd_transformed.paint_uniform_color([1, 0, 0])  # Red - transformed source
    target_pcd.paint_uniform_color([0, 1, 0])  # Green - target
    
    o3d.visualization.draw_geometries([source_pcd_transformed, target_pcd],
                                      window_name="ICP Registration Result")
    
    print("Done!")

if __name__ == "__main__":
    main()