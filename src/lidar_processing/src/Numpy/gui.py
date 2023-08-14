import cv2
import sys

## ROS STUFF
sys.path.append('/usr/lib/python2.7/dist-packages')
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import rospy #/ /home/c66tang/anaconda3/lib/python3.7/site-packages:/home/c66tang/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages
import ros_numpy
import numpy as np
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from sensor_msgs.msg import CompressedImage, PointCloud2
from cv_bridge import CvBridge
from std_msgs.msg import Float32

x_min = Float32()
x_max = Float32()
y_min = Float32()
y_max = Float32()
z_min = Float32()
z_max = Float32()

def update_xmin(msg):
    x_min.data = float(msg)/4 - 5
def update_xmax(msg):
    x_max.data = float(msg)/4 - 5
def update_ymin(msg):
    y_min.data = float(msg)/4 - 5
def update_ymax(msg):
    y_max.data = float(msg)/4 - 5
def update_zmin(msg):
    z_min.data = float(msg)/4 - 5
def update_zmax(msg):
    z_max.data = float(msg)/4 - 5

# -2 : 2 x
# -1 : 2 y
# Any

# win_name = "GUI"
win_name = "Projected Points"

cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
cv2.waitKey(1)

cv2.createTrackbar('x_min', win_name, 0, 80, update_xmin)# self.update_val_roll)
cv2.createTrackbar('x_max', win_name, 80, 80, update_xmax)# self.update_val_pitch)
cv2.createTrackbar('y_min', win_name,0, 80, update_ymin)# self.update_val_yaw)
cv2.createTrackbar('y_max', win_name, 80, 80, update_ymax)# self.update_val_tx)
cv2.createTrackbar('z_min', win_name, 0, 80, update_zmin)# self.update_val_ty)
cv2.createTrackbar('z_max', win_name, 80, 80, update_zmax)# self.update_val_tz)

node = rospy.init_node('gui_node')

pub1 = rospy.Publisher('x_min_listener', Float32)
pub2 = rospy.Publisher('x_max_listener', Float32)
pub3 = rospy.Publisher('y_min_listener', Float32)
pub4 = rospy.Publisher('y_max_listener', Float32)
pub5 = rospy.Publisher('z_min_listener', Float32)
pub6 = rospy.Publisher('z_max_listener', Float32)

while True:
    key = cv2.waitKey(1)

    if key == ord('q'):
        exit()

    pub1.publish(x_min)
    pub2.publish(x_max)
    pub3.publish(y_min)
    pub4.publish(y_max)
    pub5.publish(z_min)
    pub6.publish(z_max)