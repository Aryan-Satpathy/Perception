import open3d

import jax.numpy as np
import ros_numpy

import rospy

import time

import typing
from typing import Any, Tuple
DbCluster = typing.Callable

class DbCluster:
    def __init__(self, epsilon = 0.3, min_points = 30, print_progress = False, log_time = True, debug = False) -> DbCluster:
        self.eps = epsilon
        self.min_pts = min_points
        self.verbosity = print_progress

        self.log_time:float = None
        if log_time:
            self.log_time = 0
        self.debug = debug

    # Cluster method
    def __call__(self, pcl: open3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.log_time is not None:
            self.log_time = time.time()
        
        # Dbscan Clustering
        labels = np.array(pcl.cluster_dbscan(eps = self.eps, min_points = self.min_pts, print_progress = self.verbosity))

        # Find out unique labels
        unique, counts = np.unique(labels, return_counts = True)

        if self.log_time is not None:
            self.log_time = time.time() - self.log_time

        if self.debug: 
            if self.log_time:
                logger = rospy.loginfo if 1 / self.log_time > 30 else rospy.logwarn
                logger(f'Time taken for clustering: {self.log_time}')
            else:
                rospy.logerr("Not logging! Set debug to False.")
        
        return labels, unique, counts

    # Display methods
    def __repr__(self) -> str:
        return f"Dbscan: eps = {self.eps}, min_points = {self.min_pts}, print_progress = {self.verbosity}"
    def __str__(self) -> str:
        return repr(self)
    
    