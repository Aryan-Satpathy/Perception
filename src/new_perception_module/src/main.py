'''
To Do:
1. Detect Ground Contact Point. (Done)
2. Projection from Camera to LIDAR. (Done)
3. Check additional filter requirement. (Required)
4. Implement same KF that was implemented for LIDAR processing. (Done)
'''

from utils import ImageListenerNode, KalmanFilter, draw_visuals

from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from geometry_msgs.msg import PoseArray, Pose

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

import torch
import time

import rospy

class PerceptionNode(ImageListenerNode):
    def __init__(self, cam_topic='/pylon_camera_node_center/image_rect/compressed', 
                 cam_info_topic='/pylon_camera_node_center/camera_info', 
                 bbox_pub_name='bbox_array_center_person', gcp_pub_name='gcp_array', 
                 debug=True):
        
        gpu = torch.device("cuda")
        cpu = torch.device("cpu")

        model = YOLO('yolov8m-pose.pt')
        model.model.to(gpu)

        self.model = model

        self.active_filters = {}
        
        super().__init__(cam_topic, cam_info_topic, bbox_pub_name, gcp_pub_name, debug)

    def process_cv_image(self, cv_image: cv2.Mat):        
        frame = cv_image.copy()

        bbox_array = BoundingBoxArray()
        gcp_array = PoseArray()
        gcp_array.header.frame_id = 'ground_aligned'

        start = time.time()
        results = self.model.track(source = frame, show = False, save = False, 
                                        conf = 0.1, iou = 0.85, persist = True)
        end = time.time()

        frame = self.process_result(results[0], bbox_array, gcp_array, frame)

        self.publish(bbox_array, gcp_array)

        if self.display:
            cv2.namedWindow("Camera Perception", cv2.WINDOW_NORMAL)
            cv2.imshow('Camera Perception', frame)
            cv2.waitKey(1)

        if self.debug:
            delta = end - start

            if delta >= 0.1:
                logger = rospy.logwarn
            else:
                logger = rospy.loginfo
            
            logger(f'Camera perception time: {round(delta * 1000, 2)} ms')
        
        return cv_image
    
    def process_result(self, result: Results, bbox_array: BoundingBoxArray,
                       gcp_array: PoseArray, frame: cv2.Mat):
        
        for i in range(result.boxes.shape[0]):
            box = result.boxes[i]
            keypoint = result.keypoints[i]
            
            xll, yll, xur, yur = tuple(map(int, box.xyxy[0]))

            bbox = BoundingBox()
            bbox.pose.position.x = xll
            bbox.pose.position.y = yll
            bbox.dimensions.x = abs(xll - xur)
            bbox.dimensions.y = abs(yll - yur)
            
            conf = box.conf[0]   
            _class = box.cls[0]    
            _id = box.id[0]

            bbox.value = _id
            bbox_array.boxes.append(bbox)

            # 2 foot points
            x1, y1 = float(keypoint[-1, 0]), float(keypoint[-1, 1])
            x2, y2 = float(keypoint[-2, 0]), float(keypoint[-2, 1])

            # X, Y in Image Frame
            u = (x1 + x2) / 2
            v = (y1 + y2) / 2

            uv = np.array([u, v, 1])

            new_u_m = self.info.final_matrix[:, :-1].copy()
            new_u_m[:, -1] = self.info.final_matrix[:, -2] * 2.75 + self.info.final_matrix[:, -1]

            pos3d = np.linalg.pinv(new_u_m).dot(uv)
            pos3d = pos3d * (1 / pos3d[-1])
            pos3d[-1] = 2.75
            pos3d[0] *= -1 # I dont know why I need this. I just do.

            # Potential Filter Code Goes Here.
            kf = self.active_filters.get(_id, None)
            if kf is None:
                kf = KalmanFilter(pos3d[: -1], dt = 0.1, id = _id)
            
            kf.update(pos3d[: -1])
            pos3d = kf.predict()
            pos3d[-1] = 2.75
            self.active_filters[_id] = kf

            gcp = Pose()
            gcp.position.x, gcp.position.y, gcp.position.z = tuple(pos3d)
            
            gcp_array.poses.append(gcp)

            frame = draw_visuals(frame, (xll, yll, xur, yur), (int(u), int(v)), _class, _id)
        
        return frame

def main():
    node = PerceptionNode()

    try:
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)


if __name__ == '__main__':
    main()