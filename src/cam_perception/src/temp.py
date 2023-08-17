import sys
import time

import numpy as np
import os

from PIL import Image as PILImage

import cv2

## ROS STUFF
# sys.path.append('/usr/lib/python2.7/dist-packages')
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

import rospy #/ /home/c66tang/anaconda3/lib/python3.7/site-packages:/home/c66tang/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

class ImageListenerNode:

    def __init__(self, topic_name):
        rospy.init_node('image_feed_to_deepsort')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic_name, CompressedImage, self.image_callback)
        # self.image_sub = rospy.Subscriber(topic_name, Image, self.image_callback)
        self.last_frame = None

        # self.image_pub = rospy.Publisher('Undistorted', Image)

        camera_matrix = np.array([1510.662906, 0.000000, 949.632566, 0.000000, 1509.789449, 532.263988, 0.000000, 0.000000, 1.000000], np.float64).reshape((3, 3))
        distortion_coefficients = np.array([-0.366143, 0.140569, 0.000135, -0.002298, 0.000000], np.float64)

        w,  h = 1920, 1200
        newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w,h), 1, (w,h))

        # undistort
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, newcameramtx, (w,h), 5)

        self.out = cv2.VideoWriter('UndistortedView.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 20, (992, 1782)[: : -1])

        self.display = True
        time.sleep(1)

    def image_callback(self, msg:Image):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)

            # print(np_arr.shape)

            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # cv_image = self.bridge.imgmsg_to_cv2(msg)

            # msg.data.

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
        self.last_frame = cv2.remap(cv_image, self.mapx, self.mapy, cv2.INTER_LINEAR)
        
        x, y, w, h = self.roi
        self.last_frame = self.last_frame[y:y+h, x:x+w]

        # # Convert the NumPy array to a PIL Image
        # pil_image = PILImage.fromarray(self.last_frame)

        # # Convert the PIL Image to a ROS Image message
        # ros_image = Image()
        # ros_image.header.stamp = rospy.Time.now()
        # ros_image.height = pil_image.height
        # ros_image.width = pil_image.width
        # ros_image.encoding = "rgb8"  # Adjust the encoding as needed
        # ros_image.step = ros_image.width * 3
        # ros_image.data = np.array(pil_image).tobytes()
        print(self.last_frame.shape)
        self.out.write(self.last_frame)
        
        # try:
        #     image_msg = self.bridge.cv2_to_imgmsg(self.last_frame)
        # except Exception as e:
        #     print(e)

        # self.image_pub.publish(ros_image)

        dst = cv2.resize(self.last_frame, None, fx = 0.7, fy = 0.7)
        cv2.imshow('Feed', dst)
        key = cv2.waitKey(2)

        # if key == ord('q'):
        #     self.out.release()
        #     time.sleep(2)
        #     exit()

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

def main():
    il = ImageListenerNode('/pylon_camera_node_center/image_rect/compressed')
    # il = ImageListenerNode('/Undistorted')
    
    rospy.spin()

    il.out.release()
    
if __name__ == '__main__':
    main()