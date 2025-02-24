# CAUTION: This is NOT a stand alone script, go through the README file for details
# Author: Antar Mazumder
# Email: antar_mazumder@mines.edu
# Description: This script processes images to generate depth maps, 
# converts them into point clouds, and creates meshes using depth estimation.
# The point clouds and meshes are saved in multiple formats such as .ply, .pcd, and .xyz.
# Dependencies: Open3D, NumPy, PyTorch, Pillow, OpenCV, DepthAnythingV2

import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch

from depth_anything_v2.dpt import DepthAnythingV2

# Configuration
CONFIG = {
    'encoder': 'vitb',  
    'load_from': 'Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth',
    'max_depth': 200,  # Increased for astronomical scale
    'img_path': 'Depth-Anything-V2/mars.png',
    'outdir': './vis_pointcloud',
    'focal_length_x': 470.4,
    'focal_length_y': 470.4,
    'save_formats': ['ply', 'pcd', 'xyz'],  # Multiple formats for compatibility
    'remove_outliers': True,  # Option to remove noisy points
    'voxel_size': 0.006  # For downsampling dense areas
}

def create_mesh_from_points(pcd, depth_values, width, height):
    """Convert point cloud to mesh using surface reconstruction"""
    # First ensure we have normals
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))

    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=9,  # Octree depth, controls detail level
        width=0,  # Set to 0 for automatic width estimation
        scale=1.1,  # Scale factor to ensure holes are closed
        linear_fit=False  # Use non-linear optimization
    )

    # Remove low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Optional: Cleanup mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    return mesh

def process_point_cloud(pcd, remove_outliers=True, voxel_size=0.05):
    """Process point cloud to improve quality"""
    # Remove statistical outliers if enabled
    if remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Voxel downsampling to reduce density in crowded areas
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Estimate normals for better visualization
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    return pcd

def save_point_cloud(pcd, base_path, formats=['ply']):
    """Save point cloud in multiple formats"""
    for format in formats:
        out_path = f"{base_path}.{format}"
        if format == 'ply':
            o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
        elif format == 'pcd':
            o3d.io.write_point_cloud(out_path, pcd)
        elif format == 'xyz':
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            normals = np.asarray(pcd.normals)
            with open(out_path, 'w') as f:
                for i in range(len(points)):
                    x, y, z = points[i]
                    r, g, b = colors[i]
                    nx, ny, nz = normals[i]
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f} {nx:.6f} {ny:.6f} {nz:.6f}\n")
        print(f"Saved point cloud as: {out_path}")

def save_mesh(mesh, base_path, formats=['ply']):
    """Save mesh in multiple formats"""
    for format in formats:
        out_path = f"{base_path}_mesh.{format}"
        if format == 'ply':
            o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=True)
        elif format == 'obj':
            o3d.io.write_triangle_mesh(out_path, mesh)
        print(f"Saved mesh as: {out_path}")

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[CONFIG['encoder']])
    depth_anything.load_state_dict(torch.load(CONFIG['load_from'], map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Fix: Define filenames before the loop
    if os.path.isfile(CONFIG['img_path']):
        filenames = [CONFIG['img_path']]
    else:
        filenames = glob.glob(os.path.join(CONFIG['img_path'], '**/*'), recursive=True)
        filenames = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # Filter for images

    if not filenames:
        print(f"No valid image files found in {CONFIG['img_path']}")
        return

    os.makedirs(CONFIG['outdir'], exist_ok=True)

    for k, filename in enumerate(filenames):
        print(f'Processing {k+1}/{len(filenames)}: {filename}')

        try:
            color_image = Image.open(filename).convert('RGB')
            width, height = color_image.size

            image = cv2.imread(filename)
            if image is None:
                print(f"Failed to load image: {filename}")
                continue

            # Get depth prediction
            pred = depth_anything.infer_image(image, height)
            
            # Scale depth to max_depth and apply gamma correction for better depth detail
            pred = (pred / pred.max()) * CONFIG['max_depth']
            pred = np.power(pred / CONFIG['max_depth'], 0.8) * CONFIG['max_depth']  # Gamma correction

            resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)

            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x = (x - width / 2) / CONFIG['focal_length_x']
            y = (y - height / 2) / CONFIG['focal_length_y']
            z = np.array(resized_pred)

            # Filter out background points
            mask = z > (z.max() * 0.1)  # Remove very distant points
            x = x[mask]
            y = y[mask]
            z = z[mask]
            colors = np.array(color_image)[mask]

            points = np.stack((np.multiply(x, z.flatten()), 
                             np.multiply(y, z.flatten()), 
                             z.flatten()), axis=-1)
            colors = colors.reshape(-1, 3) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Process the point cloud
            pcd = process_point_cloud(pcd, 
                                    remove_outliers=CONFIG['remove_outliers'],
                                    voxel_size=CONFIG['voxel_size'])

            # Create mesh from point cloud
            mesh = create_mesh_from_points(pcd, pred, width, height)

            # Save both point cloud and mesh
            base_path = os.path.join(CONFIG['outdir'], 
                                   os.path.splitext(os.path.basename(filename))[0])
            save_point_cloud(pcd, base_path, CONFIG['save_formats'])
            save_mesh(mesh, base_path, ['ply', 'obj'])

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

if __name__ == '__main__':
    main()
