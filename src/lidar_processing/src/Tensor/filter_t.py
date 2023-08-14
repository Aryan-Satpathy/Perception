from typing import Any
import numpy as np

import torch

import open3d as o3d

def compute_voxel_size(points: torch.Tensor) -> torch.Tensor:
    # Custom function to define voxel size based on radial distance
    
    # # Increasing with Radial Distance
    # return 0.01 * radial_distances

    # # Decreasing with Radial Distance
    # return 0.01 * np.exp(-np.power(radial_distances, 0.5))

    pts = points.clone()

    pts[:, 0] = torch.abs(pts[:, 0])

    # If points have x in some bounds, they have specific voxel_sizes
    voxel_sizes = torch.ones((pts.shape[0], 1)) * 0.05
    
    indexes_lbound = torch.where(torch.all(pts >= [3, -torch.inf, -torch.inf], axis = -1))
    voxel_sizes[indexes_lbound] = voxel_sizes[indexes_lbound] * 0.2
    
    return voxel_sizes

class ROI:
    def __init__(self, lb: tuple, ub: tuple) -> None:
        
        self.X_MIN, self.Y_MIN, self.Z_MIN = lb
        self.X_MAX, self.Y_MAX, self.Z_MAX = ub

        self.ub_t = torch.Tensor([self.X_MAX, self.Y_MAX, self.Z_MAX,  torch.inf])
        self.lb_t = torch.Tensor([self.X_MIN, self.Y_MIN, self.Z_MIN, -torch.inf])

    # Filter method
    def __call__(self, points_np: torch.Tensor) -> torch.Tensor:
        indexes_ubound = points_np <= self.ub_t
        indexes_lbound = points_np >= self.lb_t 

        indexes_instersection = torch.where(torch.all(torch.bitwise_and(indexes_lbound, indexes_ubound), axis = -1))

        return points_np[indexes_instersection]

    # Display methods
    def __repr__(self) -> str:
        return f"ROI: {self.X_MIN} - {self.X_MAX} | {self.Y_MIN} - {self.Y_MAX} | {self.Z_MIN} - {self.Z_MAX}"
    def __str__(self) -> str:
        return repr(self)
    
class Downsample:
    def __init__(self, vox_size_dense: float = 0.05, vox_size_sparse: float = 0.01, x_lim: float = 3) -> None:
        self.vox_size_dense = vox_size_dense
        self.vox_size_sparse = vox_size_sparse
        self.x_lim = x_lim

        self.x_lim_tensor = torch.Tensor([self.x_lim, torch.inf, torch.inf, torch.inf])

    # Filter method
    def __call__(self, points_np: torch.Tensor) -> o3d.geometry.PointCloud:
        ## Version 2 (Faster than Version 1 but why do we have to convert numpy array to a list)
        
        pts = points_np.clone()
        pts[:, 0] = torch.abs(pts[:, 0])

        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points_np[:, :-1])
        pcl.colors = o3d.utility.Vector3dVector(torch.reshape(torch.Tensor.repeat(points_np[:, -1].unsqueeze(-1), 1, 1, 3), (points_np.shape[0], 3)))

        # Distinguish between dense and sparse regions
        indexes_dense = list(torch.where(torch.all(pts < self.x_lim_tensor, axis = -1))[0])
        
        dense_pcl = pcl.select_by_index(indexes_dense)
        sparse_pcl = pcl.select_by_index(indexes_dense, invert = True)

        dense_pcl = dense_pcl.voxel_down_sample(self.vox_size_dense)
        sparse_pcl = sparse_pcl.voxel_down_sample(self.vox_size_sparse)

        pcl = dense_pcl + sparse_pcl
        return pcl

        ## Version 1
        
        # indexes_dense = np.where(np.all(pts < [self.x_lim, np.inf, np.inf, np.inf], axis = -1))
        # indexes_sparse = np.where(np.all(pts >= [self.x_lim, -np.inf, -np.inf, -np.inf], axis = -1))
        # dense_pts = points_np[indexes_dense]
        # sparse_pts = points_np[indexes_sparse]

        # dense_pcl = o3d.geometry.PointCloud()
        # dense_pcl.points = o3d.utility.Vector3dVector(dense_pts[:, :-1])
        # dense_pcl.colors = o3d.utility.Vector3dVector(np.reshape(np.repeat(dense_pts[:, -1], 3, axis = 0), (dense_pts.shape[0], 3)))

        # sparse_pcl = o3d.geometry.PointCloud()
        # sparse_pcl.points = o3d.utility.Vector3dVector(sparse_pts[:, :-1])
        # sparse_pcl.colors = o3d.utility.Vector3dVector(np.reshape(np.repeat(sparse_pts[:, -1], 3, axis = 0), (sparse_pts.shape[0], 3)))

        # dense_pcl = dense_pcl.voxel_down_sample(self.vox_size_dense)
        # sparse_pcl = sparse_pcl.voxel_down_sample(self.vox_size_sparse)

        # pcl = dense_pcl + sparse_pcl
        # return pcl
    
    # Display methods
    def __repr__(self) -> str:
        return f"Voxel Downsampler: |x| >= {self.x_lim} = {self.vox_size_sparse}, |x| < {self.x_lim} = {self.vox_size_dense}"
    def __str__(self) -> str:
        return repr(self)