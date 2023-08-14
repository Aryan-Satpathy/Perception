from ultralytics import YOLO
from ultralytics.yolo.utils import ops
import torch
import sys
import time

from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import torch.nn as nn
import torch.functional as F

from ultralytics.yolo.utils.metrics import bbox_iou
from ultralytics.yolo.utils.tal import bbox2dist

import numpy as np
import os

from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from geometry_msgs.msg import PoseArray, Pose

import rospy #/ /home/c66tang/anaconda3/lib/python3.7/site-packages:/home/c66tang/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

# cudnn.benchmark = True

class ImageListenerNode:

    def __init__(self, topic_name, bbox_pub_name = 'bbox_array_center_person', gcp_pub_name = 'gcp_array'):
        rospy.init_node('image_feed_to_deepsort')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic_name, CompressedImage, self.image_callback)
        self.bbox_pub = rospy.Publisher(bbox_pub_name, BoundingBoxArray)
        self.gcp_pub = rospy.Publisher(gcp_pub_name, PoseArray)
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
        self.last_frame = cv_image

        # if self.display:
        #     cv2.imshow('Feed', self.last_frame)
        # if cv2.waitKey(2) == ord('n'):
        #     self.display = not self.display

        # Example: Print the shape of the cv_image
        # print(cv_image.shape)

    def publish(self, bbox_msg: BoundingBoxArray, gcp_msg: PoseArray):
        self.bbox_pub.publish(bbox_msg)
        gcp_msg.header.frame_id = 'ground_aligned'
        self.gcp_pub.publish(gcp_msg)
        print('Published')

    def get_width(self):
        if self.last_frame is not None:
            return self.last_frame.shape[1]
        else: return 0

    def get_height(self):
        if self.last_frame is not None:
            return self.last_frame.shape[0]
        else: return 0

class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)

class CustomDataset(Dataset):
    def __init__(self, train_directory, batch_size = 1):
        self.path = train_directory
        self.file_names = list(os.listdir(train_directory))
        self.image_names = [file_name for file_name in self.file_names if file_name[-1] == 'g']
        self.label_names = [img_name.split('.jpg')[0] + '.txt' for img_name in self.image_names]
        self.batch_size = batch_size
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, index):
        img_name, label_name = self.image_names[index], self.label_names[index]

        img = torchvision.io.read_image(self.path + img_name)
        img = img[None, :]
        
        try:
            with open(self.path + label_name, 'r') as f:
                unprocessed_labels = f.readlines()
        
            labels = [tuple(map(eval, line[:-1].split(' '))) for line in unprocessed_labels]
        except:
            labels = []
        
        return img, labels

def train():
    epochs = 5
    checkpoint_interval = 1
    checkpoint_dir = r'../models/'

    gpu = torch.device("cuda")

    model = YOLO('yolov8m.pt')
    model.model.to(gpu)
    
    model.train(model = 'yolov8m.yaml',data='coco128.yaml', epochs=1, imgsz=640, pretrain = 'yolov8m.pt')
    exit()
    path_to_data = '../data/'
    
    train_dataset = CustomDataset(path_to_data + r'train/')
    test_dataset = CustomDataset(path_to_data + r'test/')

    Tensor = torch.cuda.FloatTensor if gpu else torch.FloatTensor

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.model.parameters()))

    for epoch in range(epochs):
        for batch_i, (imgs, targets) in enumerate(train_dataset):
            # for imgs, targets in enumerate(train_dataset):
            imgs = Variable(imgs.type(Tensor))
            
            optimizer.zero_grad()

            result = model.predict(imgs)[0]
            
            x1, y1, x2, y2 = tuple(map(int, result.boxes.xyxy[0]))   # box with xyxy format, (N, 4)
            _class = result.boxes.cls[0]    # cls, (N, 1)



            exit()

            loss.backward()
            optimizer.step()

            print(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch,
                    epochs,
                    batch_i,
                    len(train_dataset),
                    model.losses["x"],
                    model.losses["y"],
                    model.losses["w"],
                    model.losses["h"],
                    model.losses["conf"],
                    model.losses["cls"],
                    loss.item(),
                    model.losses["recall"],
                    model.losses["precision"],
                )
            )

            model.seen += imgs.size(0)

        if epoch % checkpoint_interval == 0:
            model.save_weights("%s/%d.weights" % (checkpoint_dir, epoch))

def main():
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")

    model = YOLO('yolov8m-pose.pt')

    # model.model.to(gpu)
    model.model.to(gpu)

    # model.info()  # display model information

    il = ImageListenerNode('/pylon_camera_node_center/image_rect/compressed')

    r_matrix = np.array([[-0.9205, -0.0416, 0.3885, 0],
                                 [0.0415, -0.9991, -0.0087, 0],
                                  [0.3886, 0.0082, 0.9214, 0],
                                   [0, 0, 0, 1]])
    
    t_matrix = np.array([[1, 0, 0, -0.0086],
                         [0, 1, 0, 0.0491],
                         [0, 0, 1, -0.1610],
                         [0, 0, 0, 1]])
    
    extrinsic_matrix = t_matrix @ r_matrix
    cam_lidar = extrinsic_matrix# np.linalg.pinv(extrinsic_matrix)

    ext_matrix = np.array([[-0.918597, -0.0510159, 0.391888, 0.060849],
                                        [0.049842, -0.99867, -0.0131756, 0.07408], 
                                        [0.392039, 0.00742942, 0.919919, 0.151567],
                                        [0, 0, 0, 1]])
        
    ultimate_proj_matrix = np.array([850.901015, 0.0, 984.902536, 0.0,
                                        0.0, 851.293451, 600.492732, 0.0,
                                        0.0, 0.0, 1.0, 0.0]).reshape((3, 4))
    
    ultimate_proj_matrix = ultimate_proj_matrix.dot(np.linalg.pinv(ext_matrix))

    # camera_matrix = np.array([1510.662906, 0.000000, 949.632566, 0.000000, 1509.789449, 532.263988, 0.000000, 0.000000, 1.000000], np.float64).reshape((3, 3))
    # distortion_coefficients = np.array([-0.366143, 0.140569, 0.000135, -0.002298, 0.000000], np.float64)

    # h,  w = il.get_height(), il.get_width()
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w,h), 1, (w,h))

    # undistort
    # mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, newcameramtx, (w,h), 5)

    while True:
        frame = np.copy(il.last_frame)

        dst = frame
        # dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        # crop the image
        # x, y, w, h = roi
        x, y, w, h = 0, 0, dst.shape[0] - 1, dst.shape[1] - 1
        dst = dst[y:y+h, x:x+w]
        # frame = cv2.imread(r'image.png', cv2.IMREAD_COLOR)
        # frame = torch.as_tensor([frame])
        # frame = frame.permute(0, 3, 2, 1)
        # frame.to(gpu)

        bbox_array = BoundingBoxArray()
        gcp_array = PoseArray()
        gcp_array.header.frame_id = 'ground_aligned'

        start = time.time()
        results = model.track(source = dst, show = False, save = False, conf = 0.1, iou = 0.85, persist = True)
        end = time.time()

        # print(dir(results[0].boxes))
        # exit()

        print(f'Final time required: {end - start}')

        # for result in results:
        #     try: 
        #         # Detection
        #         x1, y1, x2, y2 = tuple(map(int, result.boxes.xyxy[0]))   # box with xyxy format, (N, 4)
        #         # result.boxes.xywh   # box with xywh format, (N, 4)
        #         # result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
        #         # result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
        #         conf = result.boxes.conf[0]   # confidence score, (N, 1)
        #         _class = result.boxes.cls[0]    # cls, (N, 1)
        #         _id = result.boxes.id

        #         frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 4, lineType=cv2.LINE_AA)
        #         frame = cv2.putText(frame, f"{_id}", (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 100), 2, lineType=cv2.LINE_AA)
        #     except:
        #         pass
        # rospy.logerr(f"Number of Results: {len(results)}")
        # print(results[0].boxes.xyxy)
        # exit()
        result = results[0]
        #keypoints = result.keypoints
        # for box in result.boxes:
        for i in range(result.boxes.shape[0]):
            box = result.boxes[i]
            keypoint = result.keypoints[i]
            
            try: 
                # Detection
                # print(len(result.boxes.xyxy))
                # rospy.logerr(f"Number of boxes in a result({i}): {len(result.boxes)}")
                # x1, y1, x2, y2 = tuple(map(int, result.boxes.xyxy[0]))   # box with xyxy format, (N, 4)

                # print(dir(box))
                x1, y1, x2, y2 = tuple(map(int, box.xyxy[0]))

                bbox = BoundingBox()
                bbox.pose.position.x = x1
                bbox.pose.position.y = y1
                bbox.dimensions.x = abs(x1 - x2)
                bbox.dimensions.y = abs(y1 - y2)
                
                # result.boxes.xywh   # box with xywh format, (N, 4)
                # result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
                # result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
                conf = box.conf[0]   # confidence score, (N, 1)
                _class = box.cls[0]    # cls, (N, 1)
                _id = box.id[0]

                bbox.value = _id
                bbox_array.boxes.append(bbox)

                dst = cv2.rectangle(dst, (x1, y1), (x2, y2), (255, 255, 0), 4, lineType=cv2.LINE_AA)
                dst = cv2.putText(dst, f"{_id}", (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 255, 100), 2, lineType=cv2.LINE_AA)
                
                # 2 foot points
                x1, y1 = float(keypoint[-1, 0]), float(keypoint[-1, 1])
                x2, y2 = float(keypoint[-2, 0]), float(keypoint[-2, 1])

                # X, Y in Image Frame
                u = (x1 + x2) / 2
                v = (y1 + y2) / 2

                dst = cv2.circle(dst, (int(u), int(v)), 10, (200, 255, 100), -1, cv2.LINE_AA)

                uv = np.array([u, v, 1])

                print(ultimate_proj_matrix.shape)

                # out = ultimate_proj_matrix.dot(np.array([u, v, 1]))
                # w = -2.75 / (ultimate_proj_matrix[2, :].dot(np.array([u, v, 1])))
                # # w = (2.75 - ultimate_proj_matrix[2, 0] * u - ultimate_proj_matrix[2, 1] * v) / ultimate_proj_matrix[2, 2]

                # # d = 2.75 / cam_lidar[2, 2] - (cam_lidar[2, 0] * x + cam_lidar[2, 1] * y + cam_lidar[2, 3]) / cam_lidar[2, 2]
                # # pos3d = cam_lidar @ np.array([x, y, d, 1])
                # pos3d = ultimate_proj_matrix.dot(np.array([w * u, w * v, w]))
                # # pos3d = ultimate_proj_matrix.dot(np.array[u, v, w])

                new_u_m = ultimate_proj_matrix[:, :-1].copy()
                new_u_m[:, -1] = ultimate_proj_matrix[:, -2] * 2.75 + ultimate_proj_matrix[:, -1]

                pos3d = np.linalg.pinv(new_u_m).dot(uv)
                pos3d = pos3d * (1 / pos3d[-1])
                pos3d[-1] = 2.75

                _x, _y, _z = pos3d[0], pos3d[1], 2.75
                print(pos3d)

                gcp = Pose()
                gcp.position.x = -_x
                gcp.position.y = _y
                gcp.position.z = _z


                gcp_array.poses.append(gcp)
            except:
                pass

        #   camera_matrix:
        #   rows: 3
        #   cols: 3
        #   data: [1510.662906, 0.000000, 949.632566, 0.000000, 1509.789449, 532.263988, 0.000000, 0.000000, 1.000000]
        # distortion_model: plumb_bob
        # distortion_coefficients:
        # rows: 1
        # cols: 5
        # data: [-0.366143, 0.140569, 0.000135, -0.002298, 0.000000]

        # dst = cv2.resize(dst, None, fx = 0.4, fy = 0.4)
        # frame = cv2.resize(frame, None, fx = 0.4, fy = 0.4) 

        il.publish(bbox_array, gcp_array)

        # cv2.imshow('with distortion', frame)
        cv2.imshow('without distortion', dst)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()
