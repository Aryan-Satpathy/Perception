from ultralytics import YOLO
import torch
import sys
import time
import numpy as np
import os



## ROS STUFF
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

import rospy #/ /home/c66tang/anaconda3/lib/python3.7/site-packages:/home/c66tang/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
import cv2


def callback_image(msg):
    # decode compressed image into cv2 image
    np_arr = np.fromstring(msg.data, np.uint8)
    if is_compresed_image:
        current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        current_image = np_arr.reshape(msg.height, msg.width, 3)

    start = time.time()
    results = model.track(source = current_image, show = False, save = False, conf = 0.1, iou = 0.85, persist = True)
    end = time.time()

    # print(dir(results[0].boxes))
    # exit()

    print(f'Final time required: {end - start}')

    result = results[0]

    bbox_array_msg = BoundingBoxArray()
    bbox_array_msg.header = msg.header
    bbox_array_msg.boxes = []
    #keypoints = result.keypoints
    # for box in result.boxes:
    for i in range(result.boxes.shape[0]):
        box = result.boxes[i]
        #keypoint = keypoints.xy[i]
        try: 
            # Detection
            # print(len(result.boxes.xyxy))
            # rospy.logerr(f"Number of boxes in a result({i}): {len(result.boxes)}")
            # x1, y1, x2, y2 = tuple(map(int, result.boxes.xyxy[0]))   # box with xyxy format, (N, 4)
            
            # print(dir(box))
            x1, y1, x2, y2 = tuple(map(int, box.xyxy[0]))
            
            # result.boxes.xywh   # box with xywh format, (N, 4)
            # result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
            # result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
            conf = box.conf[0]   # confidence score, (N, 1)
            _class = box.cls[0]    # cls, (N, 1)
            _id = box.id[0]

            ibox = BoundingBox()
            ibox.header = msg.header
            ibox.pose.position.x = x1
            ibox.pose.position.y = y1
            ibox.dimensions.x = x2 - x1
            ibox.dimensions.y = y2 - y1
            ibox.value = conf
            ibox.label = 0 #0 denotes pedestrian

            bbox_array_msg.boxes.append(ibox)


            # dst = cv2.rectangle(current_image, (x1, y1), (x2, y2), (255, 255, 0), 4, lineType=cv2.LINE_AA)
            # dst = cv2.putText(current_image, f"{_id}", (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 255, 100), 2, lineType=cv2.LINE_AA)
            #for j in range(keypoint.shape[0]):
            #    key = keypoint[j]
            #    x,y = key

        except:
            pass

    # cv2.imshow('pose_detection', current_image)
    # cv2.waitKey(1)

    pub_bbox.publish(bbox_array_msg)



if __name__ == '__main__':
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")

    model = YOLO('yolov8m-pose.pt')
    # model.model.to(gpu)
    model.model.to(gpu)

    # cv2.namedWindow('pose_detection', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('pose_detection', int(1920/4), int(1200/4))

    rospy.init_node('yolo_pose_inference_node')
    is_compresed_image = rospy.get_param('~is_compresed_image', False)
    camera_topic = "/pylon_camera_node_center/image_rect"
    callback_msg_type = Image
    if is_compresed_image:
        camera_topic += "/compressed"
        callback_msg_type = CompressedImage
    rospy.Subscriber(camera_topic, callback_msg_type, callback_image, queue_size=1)
    pub_bbox = rospy.Publisher('/bbox_array_center_person', BoundingBoxArray, queue_size=1)
    rospy.spin()
