import open3d

import numpy as np
import ros_numpy

import rospy

import time

import typing
from typing import List, Tuple
GNN = typing.Callable
ICP = typing.Callable

# Debug
from sensor_msgs.msg import PointCloud2
from utils import publish_points

class Register:
    int_id_track = 0
    def __init__(self):
        self.ids = []
        self.indices = []
    
    def new(self):
        new_id = self.int_id_track
        self.int_id_track += 1
        return new_id

class SNN:
    def __init__(self, epsilon = 0.5) -> GNN:
        self.eps = epsilon ** 2

        # Debug
        self.curr_pub = rospy.Publisher('centroids/current', PointCloud2)
        self.pred_pub = rospy.Publisher('centroids/pred', PointCloud2)
        self.prev_pub = rospy.Publisher('centroids/prev', PointCloud2)

        # History
        self.last_num = 0
        self.last_centroids = []
        self.kf_trackers = []
    
    # Association method
    def __call__(self, points_np: np.ndarray, labels: np.ndarray, uniques: np.ndarray, dt: float):
        n_cluster = uniques.shape[0] - 1  # One unique label is -1 which is Outlier

        centroids = []

        for i in range(n_cluster): 
            pts = points_np[np.where(labels == i), : ][0]
            centroids.append(np.mean(pts, 0))
        
        # Debug
        if self.last_num:
            self.prev_pub.publish(publish_points(self.last_centroids))

        self.last_centroids = [self.kf_trackers[i].predict() for i in range(self.last_num)]

        # Debug
        self.curr_pub.publish(publish_points(centroids))
        if self.last_num:
            self.pred_pub.publish(publish_points(self.last_centroids))

        pairs = []
        _ = range(self.last_num)
        
        for i in range(n_cluster):
            centroid_now = centroids[i]
            curr_list = []

            closest = None if self.last_num == 0 else min(_, key = lambda x: 
                           (centroid_now - self.last_centroids[x])[:-1].T @ (centroid_now - self.last_centroids[x])[:-1])
            
            # Debug statement, remove
            if self.last_num > 0:
                if (centroid_now - self.last_centroids[closest])[:-1].T @ (centroid_now - self.last_centroids[closest])[:-1] > self.eps:
                    print('Greater than eps')
            else:
                print('Last num = 0')

            pairs.append(closest
                         if self.last_num > 0 and (centroid_now - self.last_centroids[closest])[:-1].T @ (centroid_now - self.last_centroids[closest])[:-1] < self.eps
                         else None)

        return centroids, pairs
    
    # Update method
    def update(self, final_num: int, final_centroids: List[np.ndarray], kf_trackers: List[np.ndarray]):
        self.last_num = final_num
        self.last_centroids = final_centroids
        self.kf_trackers = kf_trackers

    # Display methods
    def __repr__(self):
        return f"SNN: epsilon = {self.eps ** 0.5}"
    def __str__(self) -> str:
        return repr(self)
    
class KalmanFilter:
    def __init__(self, init_state: np.ndarray, dt: float,
                 process_noise_scale: float = 0.2, measurement_noise_scale: float = 1.3, id: None = None):
        
        self.state = np.array([init_state[0], 0, 0, 0, init_state[1], 0, 0, 0])

        self.StateTransition = np.array([[1, dt, 0.5 * dt ** 2, 0.33 * dt ** 3, 0, 0, 0, 0],
                                                  [0,  1,            dt,  0.5 * dt ** 2, 0, 0, 0, 0],
                                                  [0,  0,             1,             dt, 0, 0, 0, 0],
                                                  [0,  0,             0,              1, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 1, dt, 0.5 * dt ** 2, 0.33 * dt ** 3],
                                                  [0, 0, 0, 0, 0,  1,            dt,  0.5 * dt ** 2],
                                                  [0, 0, 0, 0, 0,  0,             1,             dt],
                                                  [0, 0, 0, 0, 0,  0,             0,              1]])
        
        self.MeasurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 1, 0, 0, 0]])
        
        self.ProcessCovar = np.eye(8, dtype = np.float) * process_noise_scale
        
        self.MeasurementCovar = np.eye(2, dtype = np.float) * measurement_noise_scale

        self.covariance = np.eye(8)
        self.id = id
        
    def predict(self):
        self.state = np.dot(self.StateTransition, self.state)
        self.covariance = np.dot(np.dot(self.StateTransition, self.covariance), self.StateTransition.T) + self.ProcessCovar

        return np.array([self.state[0], self.state[4], 0])

    def update(self, observation: np.ndarray):
        y = observation - np.dot(self.MeasurementMatrix, self.state)
        S = np.dot(np.dot(self.MeasurementMatrix, self.covariance), self.MeasurementMatrix.T) + self.MeasurementCovar
        K = np.dot(np.dot(self.covariance, self.MeasurementMatrix.T), np.linalg.pinv(S))

        self.state += np.dot(K, y)
        self.covariance = np.dot((np.eye(8) - np.dot(K, self.MeasurementMatrix)), self.covariance)

    # Make this more readable, ugly af
    @staticmethod
    def associate_trackers(centroids: List[np.ndarray], pairs: List[int], registery: Register, snn_tracking: SNN, dt: float = 0.1):
        ids = []
        velocities = []
        kf_trackers = []
        for i in range(len(centroids)):
            closest = pairs[i]

            if closest is None: # No match
                id = registery.new()
                ids.append(id)
                registery.ids.append(id)
                registery.indices.append(i)
                velocities.append(np.zeros(centroids[0].shape))
                kf_trackers.append(KalmanFilter(centroids[i][:-1], dt = dt, id = id))
                # print(f'New because None, {id}')
            else:               # Matched with prev frame
                try: # Try excepts are very slow
                    ids.append(registery.ids[registery.indices.index(closest)])
                    registery.indices[registery.indices.index(closest)] = i
                    # velocity = (centroids[i] - self.snn_tracking.last_centroids[closest]) / self.dt
                    velocities.append(np.zeros(centroids[0].shape))
                    snn_tracking.kf_trackers[closest].update(centroids[i][:-1])
                    kf_trackers.append(snn_tracking.kf_trackers[closest])
                except:
                    id = registery.new()
                    ids.append(id)
                    registery.ids.append(id)
                    registery.indices.append(i)
                    velocities.append(np.zeros(centroids[0].shape))
                    kf_trackers.append(KalmanFilter(centroids[i][:-1], dt = dt, id = id))
                    # print(f'New because error, {id}')
                    # So there is a problem. When multiple centroids get associated to same centroid
                    # from previous frame, they override. 
        
        return ids, kf_trackers
        
class ICP:
    # Not complete. Do not look for meaning in the code
    def __init__(self, epsilon) -> ICP:
        self.eps = epsilon

        # History
        self.last_points_np: np.ndarray = None
        self.last_labels: np.ndarray = None
        self.last_unique: np.ndarray = None

    # Association method
    def __call__(self, points_np: np.ndarray, labels: np.ndarray, uniques: np.ndarray, pairs: List[list]):
        n_cluster = uniques.shape[1] - 1  # One unique label is -1 which is Outlier

        for i in range(n_cluster):
            associated_indices = pairs[i]
            
            target_pts = points_np[np.where(labels == i)]
            target_cld = open3d.geometry.PointCloud()
            target_cld.points = open3d.utility.Vector3dVector(target_pts)

            for j in associated_indices:
                source_pts = self.last_points_np[np.where(self.last_labels == j)]
                source_cld = open3d.geometry.PointCloud()
                source_cld.points = open3d.utility.Vector3dVector(source_pts)

                init_transform = None

                reg_p2p = open3d.pipelines.registration.registration_icp(
                    source_cld, target_cld,
                    self.eps, init_transform
                )
                