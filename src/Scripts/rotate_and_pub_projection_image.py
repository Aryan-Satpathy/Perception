#!/usr/bin/env python
import rospy
import numpy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

bridge = CvBridge()

# Image callback where input image is rotated 180 degrees
def callback_Image(ros_image):
    # image = bridge.compressed_imgmsg_to_cv2(ros_image, desired_encoding="bgr8")
    image = bridge.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    image_message = bridge.cv2_to_imgmsg(rotated_image)
    img_pub.publish(image_message)


if __name__ == '__main__':
    global img_pub
    rospy.init_node('rotate_image', anonymous=True)

    img_pub = rospy.Publisher("/projection_img/flipped",Image,queue_size=1)

    rospy.Subscriber("/projection_img", Image, callback_Image)
    
    rospy.spin()