#!/usr/bin/env python3

'''
To Do:
1. Late fusion of LIDAR and Camera Predictions: Done
2. Add Multi Camera Support: Done
3. Do not make it LIDAR/Camera Centric: How2 :(
'''

from munkres import Munkres
from utils import ListenerNode, Detection

import rospy
import rosparam

import cv2
import numpy as np

PARAM_PREFIX = '/lidar_cam_fusion_py_node/'

class Node(ListenerNode):
    def __init__(self, num_cameras: int = 1, threshold: float = 1e-2):
        self.munkres = Munkres()
        self.threshold = threshold
        self.Detections = None

        self.caller = rospy.Timer(period= 0.1, callback = self.fuse)

        super().__init__(num_cameras)
    
    def fuse(self):
        cost_matrix = self.generate_cost_matrices()

        if cost_matrix is None:
            # Premature callback. Wait for LiDAR and Camera Perception to be ready.
            return

        indexes = [self.munkres.compute(cost_matrix[i]) for i in range(self.num_cameras)]

        lidar_asssociations = [[None for j in range(self.num_cameras)] for i in range(len(self.lidar_ids))]

        for cam_id in range(self.num_cameras):
            for l_d, c_d in indexes[cam_id]:
                cost = cost_matrix[cam_id][l_d][c_d]

                if cost > self.threshold:
                    continue
                
                lidar_asssociations[l_d][cam_id] = c_d
            
        Detections = [Detection([(None if j is None else self.cam_ids[j]) for j in lidar_asssociations[i]], self.lidar_ids[i], dummy = True) for i in range(len(self.lidar_ids))]

        if self.Detections is not None:
            for detection in Detections:
                for _detection in self.Detections:
                    if detection == _detection:
                        detection.update(_detection)
                        break
                else:
                    detection.assign()

        self.Detections = Detections

        # for row, column in indexes:
        #     value = cost_matrix[row][column]
        #     total += value
        #     print(f'({row}, {column}) -> {value}')
        
        # print(f'Total association cost = {total}.')

        rospy.logwarn(f'Number of Detections so far: {Detection.count}')

        return super().fuse()
    
    def generate_cost_matrices(self):
        # Incorporate 2D(Bounding Box IOU) and 3D costs(GCP eucledian distance).
        return [np.zeros(len(self.lidar_ids)) for i in range(self.num_cameras)]

def main():
    
    num_cameras = rospy.get_param(f'{PARAM_PREFIX}cameras_num')
    association_threshold = rospy.get_param(f'{PARAM_PREFIX}thresh')

    node = Node(num_cameras = num_cameras, threshold = association_threshold)

    try:
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)

if __name__ == '__main__':
    main()