import rospy
import ros_numpy

from sensor_msgs.msg import PointCloud2, CompressedImage, CameraInfo
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from geometry_msgs.msg import Pose, PoseArray

import numpy as np
import cv2

from typing import Tuple, List
import typing

Detection = typing.Collection

def array_to_pointcloud2(points, stamp=None, frame_id=None) -> PointCloud2:
    all_dtype_names = ['x','y','z','intensity','ring','timestamp']
    all_dtype_formats = ['f4','f4','f4','u1','u2','f8']
    all_dtype_offsets = [0,4,8,16,18,24]
    all_dtype_itemsizes = [4,8,12,18,24,32]
    num_fields = points.shape[1]
    
    dtype_for_points = np.dtype(
                        {'names':all_dtype_names[:num_fields],
                        'formats':all_dtype_formats[:num_fields],
                        'offsets':all_dtype_offsets[:num_fields],
                        'itemsize':all_dtype_itemsizes[num_fields-1]})
    cloud_arr = np.rec.fromarrays(points.T, dtype=dtype_for_points)
    msg = ros_numpy.point_cloud2.array_to_pointcloud2(cloud_arr, stamp, frame_id)
    msg.header.stamp = rospy.Time.now()
    return msg

def pointcloud2_to_array(raw_cloud: PointCloud2) -> np.ndarray:
    points_np_record = ros_numpy.point_cloud2.pointcloud2_to_array(raw_cloud)  # np.array(...) allows for write access to points_np_record

    # Convert np record array to np array (with just x,y,z)
    points_np = np.zeros((points_np_record['x'].flatten().shape[0], 4))
    points_np[:, 0] = points_np_record['x'].flatten()
    points_np[:, 1] = points_np_record['y'].flatten()
    points_np[:, 2] = points_np_record['z'].flatten()
    points_np[:, 3] = points_np_record['intensity'].flatten()

    return points_np

class Detection:

    c_ids = None
    l_id = None

    count = 0

    def __init__(self, c_ids: List[int], l_id: int, dummy: bool = False) -> None:
        self.c_ids = c_ids
        self.l_id = l_id
        
        if not dummy:
            self.assign()

    def update(self, other: Detection):
        self.c_ids = [other.c_ids[i] if other.c_ids[i] is not None else self.c_ids[i] for i in range(len(self.c_ids))]
        self.l_id = other.l_id if other.l_id is not None else self.l_id

        self.id = other.id

    def assign(self):
        self.id = Detection.count
        Detection.count += 1
    
    # Evaluation method
    def __eq__(self, __value: Tuple[List[int], int]) -> bool:
        c_ids, l_id = __value
        return (True in [c_ids[i] == self.c_ids[i] and (c_ids[i] is not None) for i in range(len(c_ids))]) or (self.l_id == l_id and l_id is not None)
    
    # Representation methods
    def __str__(self) -> str:
        return f'Detection <Camera_id: {self.c_id}, Camera_id: {self.l_id}, Global_id: {self.id}>'
    def __repr__(self) -> str:
        str(self)

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

class ListenerNode:
    def __init__(self, num_cameras: int = 1):
        self.node = rospy.init_node('~')

        self.num_cameras = num_cameras
        
        self.image_subs = [rospy.Subscriber(f'image_{i}', CompressedImage, lambda msg: self.image_callback(i, msg)) for i in range(self.num_cameras)]
        self.cam_info_subs = [rospy.Subscriber(f'info_{i}', CameraInfo, lambda msg: self.info_callback(i, msg)) for i in range(self.num_cameras)]
        self.cam_bbox_subs = [rospy.Subscriber(f'bbox_cam_{i}', BoundingBoxArray, lambda msg: self.cam_bbox_callback(i, msg)) for i in range(self.num_cameras)]
        self.gcp_subs = [rospy.Subscriber(f'gcp_{i}', PoseArray, lambda msg: self.gcp_callback(i, msg)) for i in range(self.num_cameras)]
        self.lidar_sub = rospy.Subscriber('lidar', PointCloud2, self.lidar_callback)
        self.lidar_bbox_sub = rospy.Subscriber('bbox_lidar', BoundingBoxArray, self.lidar_bbox_callback)

        self.info = [None for i in range(num_cameras)]
        self.last_frame = [None for i in range(num_cameras)]
        self.cam_bbox = [None for i in range(num_cameras)]
        self.cam_ids = [None for i in range(num_cameras)]
        self.gcp = [None for i in range(num_cameras)]
        self.last_pcl = None
        self.lidar_ids = None
        self.lidar_bbox = None

    def info_callback(self, id: int, msg: CameraInfo):
        try:
            info = Info()

            info.P = np.array(msg.P).reshape((3, 4))
            info.height = msg.height
            info.width = msg.width

            info.final_matrix = info.P.dot(np.linalg.pinv(info.extrinsic))

            self.info[id] = info

            self.cam_info_subs[id].unregister()
            
            if self.debug:
                rospy.loginfo(f'Recieved camera info for cam[{id}]. Ready for Cam Perception!')

        except Exception as e:
            rospy.logerr(e)
             

    def image_callback(self, id: int, msg: CompressedImage):
        if self.info[id] is None:
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            processed_image = self.process_cv_image(cv_image)

            self.last_frame[id] = processed_image

        except Exception as e:
            rospy.logerr(e)

    def process_cv_image(self, cv_image: cv2.Mat):
        # Camera Perception Goes Here
        return cv_image

    def cam_bbox_callback(self, id: int, msg: BoundingBoxArray):
        if self.info[id] is None:
            return

        try:
            self.cam_bbox[id] = [(box.pose.position.x, box.pose.position.y,
                                  box.dimensions.x, box.dimensions.y,
                                  box.value, box.label) for box in msg.boxes]
            
            self.cam_ids[id] = [box.value for box in msg.boxes]

        except Exception as e:
            rospy.logerr(e)

    def gcp_callback(self, id: int, msg: PoseArray):
        if self.info[id] is None:
            return

        try:
            self.gcp[id] = [(pose.position.x,
                             pose.position.y,
                             pose.position.z) for pose in msg.poses]

        except Exception as e:
            rospy.logerr(e)
    
    def lidar_callback(self, msg: PointCloud2):
        try:
            self.last_pcl = pointcloud2_to_array(msg)

            # Intensity channel contains indices/labels/track_ids. Find unique track_ids and you get the active tracked clusters
            self.lidar_ids = np.unique(self.last_pcl[:, -1])
        
        except Exception as e:
            rospy.logerr(e)

    def lidar_bbox_callback(self, msg: BoundingBoxArray):
        try:
            self.lidar_bbox = [(box.pose.position.x, box.pose.position.y, box.pose.position.z,
                                            box.dimensions.x, box.dimensions.y,box.dimensions.z,
                                            box.value, box.label) for box in msg.boxes]

        except Exception as e:
            rospy.logerr(e)

    def fuse(self):
        # Fusion Code Goes Here
        pass    
