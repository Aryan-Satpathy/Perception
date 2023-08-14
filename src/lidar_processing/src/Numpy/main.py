'''
Goals:
1. Filtering/ROI: Done
2. Clustering + GNN(do we need GNN?) + ICP(Why are we doing ICP again? I think its for reconstruction): Done-ish 
3. KF Tracking + ICP Tracking: Done-ish(No ICP Tracking)
4. Reconstruction: If reconstruction is slow in Python, just have a Reconstruction Node in C++. Already present in BUS code 
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

from filter import ROI, Downsample
from cluster import DbCluster
from utils import array_to_pointcloud2, pointcloud2_to_array, map_intensity, get_bbox
from track import Register, SNN, KalmanFilter

class ListenerNode:

    def __init__(self, lidar_topic = '/rslidar_points_front'):
        rospy.init_node('lidar_processor')

        self.lidar_sub = rospy.Subscriber(lidar_topic, PointCloud2, self.lidar_callback)
        self.lidar_pub = rospy.Publisher('processed_lidar', PointCloud2)
        self.bbox_pub = rospy.Publisher('processed_lidar/bbox', BoundingBoxArray)
        self.last_pcl = None

        # # Actual Setting
        # lb = (-3.75, -1.25, 1)
        # ub = (5, 2, 2.75)
        
        # Comparing Results
        lb = (-3, -0.6, 1)
        ub = (5, 1.3, 2.75)
        
        self.roi_filter = ROI(lb, ub)
        self.downsample_filter = Downsample()

        self.clustering = DbCluster(debug = False)

        self.registery = Register()
        self.snn_tracking = SNN(0.2)
        self.dt = 0
        
        self.display = False
        time.sleep(1)

    def lidar_callback(self, msg:PointCloud2):
        try:
            self.process_raw_pcl(msg)
        except Exception as e:
            rospy.logerr(e) 
    
    def process_raw_pcl(self, raw_cloud:PointCloud2):
        # Timing the function
        start_time = time.time()
        
        points_np = pointcloud2_to_array(raw_cloud = raw_cloud)
        
        # Filter based on ROI
        self.last_pcl = self.roi_filter(points_np = points_np)
        # Voxel Downsample
        pcl = self.downsample_filter(points_np = self.last_pcl)

        # Convert PointCloud to Array
        pts_array = np.asarray(pcl.points)
        self.last_pcl = np.zeros((pts_array.shape[0], pts_array.shape[1] + 1))
        self.last_pcl[:, :-1] = pts_array
        # self.last_pcl[:, -1] = np.asarray(pcl.colors)[:, -1]

        labels, unique, count = self.clustering(pcl)
        # print(count.shape[0])

        centroids, pairs = self.snn_tracking(points_np = pts_array, labels = labels, uniques = unique, dt = 0.1)

        ids, kf_trackers = KalmanFilter.associate_trackers(centroids = centroids, pairs = pairs,
                                                           registery = self.registery, 
                                                           snn_tracking = self.snn_tracking, dt = 0.1)

        self.snn_tracking.update(count.shape[0] - 1, centroids, kf_trackers)

        ids = [kf_tracker.id for kf_tracker in kf_trackers]

        # Give intensity based on Cluster Index for visualization
        f = lambda x: map_intensity(x, ids = ids, max_val = self.registery.int_id_track)
        self.last_pcl[:, -1] = np.vectorize(f)(labels).astype(points_np.dtype)
        # self.last_pcl[:, -1] = (labels / (unique.shape[0] + 1) * 255.0).astype(points_np.dtype)
    
        # Make messsage from numpy array
        processed_msg = array_to_pointcloud2(self.last_pcl, raw_cloud.header.stamp, 'ground_aligned')

        # Make message for bounding box
        bbox_msg = get_bbox(points_np = self.last_pcl[:, :-1], labels = labels, centroids = centroids,
                            ids = ids, counts = count, 
                            stamp = raw_cloud.header.stamp, frame = 'ground_aligned')
        
        # Timing the function
        end_time = time.time()
        total_time = end_time - start_time

        self.dt = total_time
        
        # Publish the message
        self.lidar_pub.publish(processed_msg)
        self.bbox_pub.publish(bbox_msg)

        # Readable Logging
        if total_time:
            logger = rospy.loginfo if (1 / total_time) > 30 else rospy.logwarn
            logger(f'Time taken for processing: {total_time}')
        else:
            rospy.logerr("Not logging! Set debug to False.")

if __name__ == '__main__':
    node = ListenerNode()

    rospy.spin()