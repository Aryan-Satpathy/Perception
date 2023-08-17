from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from geometry_msgs.msg import PoseArray, Pose

import rospy
import numpy as np
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
import cv2

import random

from typing import Tuple

class Info:
    
    P = None
    height = None
    width = None

    extrinsic = np.array([[-0.918597, -0.0510159, 0.391888, 0.060849],
                                    [0.049842, -0.99867, -0.0131756, 0.07408], 
                                    [0.392039, 0.00742942, 0.919919, 0.151567],
                                    [0, 0, 0, 1]])

    def __init__(self) -> None:
        pass

class ImageListenerNode:

    def __init__(self, cam_topic = '/pylon_camera_node_center/image_rect/compressed', 
                 cam_info_topic = '/pylon_camera_node_center/camera_info',
                 bbox_pub_name = 'bbox_array_center_person', gcp_pub_name = 'gcp_array',
                 debug = True):
        
        rospy.init_node('cam_perception')

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(cam_topic, CompressedImage, self.image_callback)
        self.info_sub = rospy.Subscriber(cam_info_topic, CameraInfo, self.info_callback)
        self.bbox_pub = rospy.Publisher(bbox_pub_name, BoundingBoxArray)
        self.gcp_pub = rospy.Publisher(gcp_pub_name, PoseArray)
        
        self.info = None
        self.last_frame = None

        self.display = True
        self.debug = debug

    def image_callback(self, msg: CompressedImage):
        if self.info is None:
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            processed_image = self.process_cv_image(cv_image)

            self.last_frame = processed_image

        except Exception as e:
            rospy.logerr(e)

    def info_callback(self, msg: CameraInfo):
        try:
            info = Info()

            info.P = np.array(msg.P).reshape((3, 4))
            info.height = msg.height
            info.width = msg.width

            info.final_matrix = info.P.dot(np.linalg.pinv(info.extrinsic))

            self.info = info

            self.info_sub.unregister()
            
            if self.debug:
                rospy.loginfo('Recieved camera info. Ready for Cam Perception!')

        except Exception as e:
            rospy.logerr(e)

    def process_cv_image(self, cv_image: cv2.Mat):
        # Process the cv_image (NumPy array) as needed
        return cv_image

    def publish(self, bbox_msg: BoundingBoxArray, gcp_msg: PoseArray):
        if self.info is None:
            return 
        
        self.bbox_pub.publish(bbox_msg)
        self.gcp_pub.publish(gcp_msg)
        
        if self.debug: rospy.loginfo('Published bbox and gcp.')

    def get_width(self):
        return self.info if self.info is None else self.info.width

    def get_height(self):
        return self.info if self.info is None else self.info.height
    
class Register:
    int_id_track = 0
    def __init__(self):
        self.ids = []
        self.indices = []
    
    def new(self):
        new_id = self.int_id_track
        self.int_id_track += 1
        return new_id
    
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
    
def draw_visuals(frame: cv2.Mat, box: Tuple[int], gcp: Tuple[int], _class: int, _id: int) -> cv2.Mat:
    _id = int(_id)

    color = color_randomizer(_id)

    box = tuple(map(int, box))
    gcp = tuple(map(int, gcp))

    frame = cv2.rectangle(frame, box[: 2], box[2: ], color, 4, cv2.LINE_AA)
    frame = cv2.circle(frame, gcp, 8, color, -1, cv2.LINE_AA)

    frame = cv2.putText(frame, f'{_id}', tuple(map(lambda x: x - 5, box[ : 2])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    
    return frame

def color_randomizer(seed = None):
    if seed is None:
        random.seed()
    else:
        random.seed(seed)
    
    color = min(255, random.randint(50, 300)), min(255, random.randint(50, 300)), min(255, random.randint(50, 300))
    
    return color