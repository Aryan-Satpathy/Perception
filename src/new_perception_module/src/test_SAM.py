from ultralytics import YOLO, SAM
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

from ultralytics.yolo.utils.metrics import bbox_iou
from ultralytics.yolo.utils.tal import bbox2dist

import numpy as np
import os

## ROS STUFF
# sys.path.append('/usr/lib/python2.7/dist-packages')

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
        self.last_frame = cv_image

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
    model.train(model = 'yolov8m.yaml',data='coco128.yaml', epochs=1, imgsz=640, pretrain = 'yolov8m.pt')
    exit()
    model.model.to(gpu)
    
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
    # cpu = torch.device("cpu")

    model = SAM('sam_b.pt')
    # model.model.to(gpu)
    model.model.to(gpu)

    # model.info()  # display model information

    il = ImageListenerNode('/pylon_camera_node_center/image_rect/compressed')

    while True:
        frame = il.last_frame
        # frame = cv2.imread(r'image.png', cv2.IMREAD_COLOR)
        # frame = torch.as_tensor([frame])
        # frame = frame.permute(0, 3, 2, 1)
        # frame.to(gpu)


        start = time.time()
        results = model.predict(source = frame, show = True, save = False)
        end = time.time()

        print(f'Final time required: {end - start}')

        for result in results:
            try: 
                # Detection
                x1, y1, x2, y2 = tuple(map(int, result.boxes.xyxy[0]))   # box with xyxy format, (N, 4)
                # result.boxes.xywh   # box with xywh format, (N, 4)
                # result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
                # result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
                conf = result.boxes.conf[0]   # confidence score, (N, 1)
                _class = result.boxes.cls[0]    # cls, (N, 1)

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 4, lineType=cv2.LINE_AA)
                frame = cv2.putText(frame, f"{_class}", (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 100), 2, lineType=cv2.LINE_AA)
            except:
                pass
        frame = cv2.resize(frame, None, fx = 0.4, fy = 0.4)

        cv2.imshow('Result', frame)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()
