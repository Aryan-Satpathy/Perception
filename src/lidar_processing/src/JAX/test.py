'''
Goals:
1. Filtering/ROI: Done
2. Clustering + GNN(do we need GNN?) + ICP(Why are we doing ICP again?): 
3. KF Tracking + ICP Tracking
4. Reconstruction
'''
import open3d as o3d

import numpy as np
import math
import time

from functools import reduce

import rospy
import ros_numpy

from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32, Float32MultiArray, Bool

from filter import ROI, compute_voxel_size
from cluster import DbCluster
from utils import array_to_pointcloud2

class ListenerNode:

    def __init__(self, lidar_topic = '/rslidar_points_front'):
        rospy.init_node('lidar_processor')

        self.lidar_sub = rospy.Subscriber(lidar_topic, PointCloud2, self.lidar_callback)
        self.lidar_pub = rospy.Publisher('processed_lidar', PointCloud2)
        self.last_pcl = None

        lb = (-3.75, -1.25, 1)
        ub = (5, 2, 2.75)
        self.roi_filter = ROI(lb, ub)

        self.clustering = DbCluster(debug = False)
        
        self.display = False
        time.sleep(1)

    def lidar_callback(self, msg:PointCloud2):
        try:
            self.process_raw_pcl(msg)
        except Exception as e:
            rospy.logerr(e) 
    
    def process_raw_pcl(self, raw_cloud:PointCloud2):
        start_time = time.time()
        
        points_np_record = ros_numpy.point_cloud2.pointcloud2_to_array(raw_cloud)  # np.array(...) allows for write access to points_np_record

        # Convert np record array to np array (with just x,y,z)
        points_np = np.zeros((points_np_record['x'].flatten().shape[0], 4))
        points_np[:, 0] = points_np_record['x'].flatten()
        points_np[:, 1] = points_np_record['y'].flatten()
        points_np[:, 2] = points_np_record['z'].flatten()
        points_np[:, 3] = points_np_record['intensity'].flatten()
        ## Not being Used points_np_intensity = points_np_record['intensity'].flatten()
        
        # Filter based on ROI
        self.last_pcl = self.roi_filter(points_np = points_np)
        
        # print(np.min(distances))
        # exit()

        # # Visualize the downsampled point cloud
        # o3d.visualization.draw_geometries([downsampled_cloud])

        # # Compute radial distance from the origin for each point
        # distances = np.linalg.norm(self.last_pcl[:, :-1], axis=1)

        # Compute voxel size for each point based on radial distance
        voxel_sizes = compute_voxel_size(self.last_pcl[:, :-1])

        voxel_assignments = {}
        for i, point in enumerate(self.last_pcl[:, :-1]):
            intensity = self.last_pcl[i, -1]
            current_voxel_size = voxel_sizes[i]
            voxel_index = tuple((point / current_voxel_size).astype(int))
            if voxel_index not in voxel_assignments:
                voxel_assignments[voxel_index] = []
            voxel_assignments[voxel_index].append((point, intensity))

        # Compute representative points for each voxel
        downsampled_points = []
        downsampled_intensities = []
        for voxel_index, point_list in voxel_assignments.items():
            points = [point for point, _ in point_list]
            intensities = [intensity for _, intensity in point_list]
            centroid = np.mean(points, axis=0)
            intensity = np.mean(intensities, axis=0)
            downsampled_points.append(centroid)
            downsampled_intensities.append(intensity)
        
        # # Create a voxel grid with varying voxel sizes
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcl, list(voxel_sizes))

        # # Downsample the point cloud using the voxel grid
        # pcl = voxel_grid.extract_voxels_point_cloud()

        # pcl = pcl.voxel_down_sample(0.05)

        downsamp_np = np.asarray(downsampled_points)
        downsamp_i_np = np.asarray(downsampled_intensities)

        self.last_pcl = np.zeros((downsamp_np.shape[0], downsamp_np.shape[1] + 1))
        self.last_pcl[:, :-1] = downsamp_np
        self.last_pcl[:, -1] = downsamp_i_np

        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(self.last_pcl[:, :-1])
        pcl.colors = o3d.utility.Vector3dVector(np.reshape(np.repeat(self.last_pcl[:, -1], 3, axis = 0), (self.last_pcl.shape[0], 3)))

        labels, unique, count = self.clustering(pcl)
        print(count.shape[0])

        self.last_pcl[:, 3] = (labels / (unique.shape[0] + 1) * 255.0).astype(points_np.dtype)
    
        processed_msg = array_to_pointcloud2(self.last_pcl, raw_cloud.header.stamp, raw_cloud.header.frame_id)
        end_time = time.time()
        total_time = end_time - start_time
        self.lidar_pub.publish(processed_msg)

        if total_time:
            logger = rospy.loginfo if (1 / total_time) > 30 else rospy.logwarn
            logger(f'Time taken for processing: {total_time}')
        else:
            rospy.logerr("Not logging! Set debug to False.")

if __name__ == '__main__':
    node = ListenerNode()

    rospy.spin()