from ultralytics import YOLO
from ultralytics.yolo.utils import ops
import torch
import sys
import time

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import torch.nn as nn
import torch.functional as F
import torchvision.transforms.functional as func

import numpy as np
import os

## ROS STUFF
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

import rospy #/ /home/c66tang/anaconda3/lib/python3.7/site-packages:/home/c66tang/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

# cudnn.benchmark = True

class ImageListenerNode:

    def __init__(self, topic_name):
        rospy.init_node('image_feed_to_deepsort')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic_name, CompressedImage, self.image_callback)
        self.last_frame = None

        self.display = True
        time.sleep(1)

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # Process the cv_image (NumPy array) as needed
            # ...
            # Return the cv_image if desired
            self.process_cv_image(cv_image)

        except Exception as e:
            rospy.logerr(e)

    def process_cv_image(self, cv_image):
        # Process the cv_image (NumPy array) as needed
        # ...
        self.last_frame = cv_image# cv2.resize(cv_image, (960, 600))

        # if self.display:
        #     cv2.imshow('Feed', self.last_frame)
        # if cv2.waitKey(2) == ord('n'):
        #     self.display = not self.display

        # Example: Print the shape of the cv_image
        # print(cv_image.shape)

    def get_width(self):
        if self.last_frame is not None:
            return self.last_frame.shape[1]
        else: return 0

    def get_height(self):
        if self.last_frame is not None:
            return self.last_frame.shape[0]
        else: return 0

class MyDataset(Dataset):
    def __init__(self, path_to_video):
        cap = cv2.VideoCapture(path_to_video)
        self.total_frames = int(cap.get(7))

        # self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.width = 960
        self.height = 600

        self.data = []# np.ones((self.total_frames, self.height, self.width, 3), np.uint8)

        for i in range(self.total_frames):
            # _, self.data[i, :, :, :] = cap.read()
            _, data = cap.read()
            data  = cv2.resize(data, (600, 960))
            self.data.append(func.to_tensor(data)) #.transpose(1,2,0)))

        cap.release()

        self._index = 0

    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index < self.total_frames:
            self._index += 1
            return self.data[self._index - 1]
        else:
            raise StopIteration     

def main():
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")

    # cap = cv2.VideoCapture('UndistortedView.avi')
    # total_frames = int(cap.get(7))

    il = ImageListenerNode('/pylon_camera_node_center/image_rect/compressed')

    model = YOLO('yolov8m-pose.pt')

    # model.model.to(gpu)
    model.model.to(gpu)

    camera_matrix = np.array([1510.662906, 0.000000, 949.632566, 0.000000, 1509.789449, 532.263988, 0.000000, 0.000000, 1.000000], np.float64).reshape((3, 3))
    distortion_coefficients = np.array([-0.366143, 0.140569, 0.000135, -0.002298, 0.000000], np.float64)

    h,  w = il.get_height(), il.get_width()
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w,h), 1, (w,h))

    x, y, _w, _h = roi

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, newcameramtx, (w,h), 5)

    codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # store = cv2.VideoWriter('ProgressMeet.avi', codec, 15, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    store = cv2.VideoWriter('ProgressMeet.avi', codec, 25, (min(_w, w - x), min(_h, h - y)))

    # model.info()  # display model information

    # camera_matrix = np.array([1510.662906, 0.000000, 949.632566, 0.000000, 1509.789449, 532.263988, 0.000000, 0.000000, 1.000000], np.float64).reshape((3, 3))
    # distortion_coefficients = np.array([-0.366143, 0.140569, 0.000135, -0.002298, 0.000000], np.float64)

    # h,  w = il.get_height(), il.get_width()
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w,h), 1, (w,h))

    # # undistort
    # mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, newcameramtx, (w,h), 5)

    # dataset = MyDataset('UndistortedView.avi')
    # dataloader = DataLoader(dataset, 1, True)

    # for frame in dataloader:
    while not rospy.is_shutdown():
        # _, 
        
        frame = np.copy(il.last_frame)

        dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        # crop the image
        x, y, w, h = roi
        frame = dst[y:y+h, x:x+w]

        if frame is None:
            break

        # dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        # frame = cv2.imread(r'image.png', cv2.IMREAD_COLOR)
        # frame = torch.as_tensor([frame])
        # frame = frame.permute(0, 3, 2, 1)
        # frame.to(gpu)

        # print(frame.shape)

        start = time.time()
        results = model.track(source = frame, show = False, save = False, conf = 0.1, iou = 0.85, persist = True)
        end = time.time()

        print(f'Final time required: {end - start}')

        result = results[0]
        
        # for box in result.boxes:
        for i in range(result.boxes.shape[0]):
            box = result.boxes[i]
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

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 255, 100), 4, lineType=cv2.LINE_AA)
                frame = cv2.putText(frame, f"{_id}", (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 255, 100), 2, lineType=cv2.LINE_AA)
            except:
                pass

        # for result in results:
        #     try: 
        #         # Detection
        #         x1, y1, x2, y2 = tuple(map(int, result.boxes.xyxy[0]))   # box with xyxy format, (N, 4)
        #         # result.boxes.xywh   # box with xywh format, (N, 4)
        #         # result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
        #         # result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
        #         conf = result.boxes.conf[0]   # confidence score, (N, 1)
        #         _class = result.boxes.cls[0]    # cls, (N, 1)

        #         dst = cv2.rectangle(dst, (x1, y1), (x2, y2), (255, 255, 0), 4, lineType=cv2.LINE_AA)
        #         dst = cv2.putText(dst, f"{_class}", (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 100), 2, lineType=cv2.LINE_AA)
        #     except:
        #         pass

        #   camera_matrix:
        #   rows: 3
        #   cols: 3
        #   data: [1510.662906, 0.000000, 949.632566, 0.000000, 1509.789449, 532.263988, 0.000000, 0.000000, 1.000000]
        # distortion_model: plumb_bob
        # distortion_coefficients:
        # rows: 1
        # cols: 5
        # data: [-0.366143, 0.140569, 0.000135, -0.002298, 0.000000]

        store.write(frame)

        # frame = cv2.resize(frame, None, fx = 0.4, fy = 0.4)
        # frame = cv2.resize(frame, None, fx = 0.4, fy = 0.4) 

        # cv2.imshow('with distortion', frame)
        cv2.imshow('Output', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    store.release()

if __name__ == '__main__':
    main()
