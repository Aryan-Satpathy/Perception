import sys
## ROS STUFF
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

import rospy #/ /home/c66tang/anaconda3/lib/python3.7/site-packages:/home/c66tang/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox

# def callback_bbox_array_chair(msg):
#     global bbox_array_person_msg
#     ## change all the label in msg to 56
#     for i in range(len(msg.boxes)):
#         msg.boxes[i].label = 56
    
#     ## add all the person bbox to msg
#     if bbox_array_person_msg is not None:
#         for i in range(len(bbox_array_person_msg.boxes)):
#             msg.boxes.append(bbox_array_person_msg.boxes[i])
#     pub.publish(msg)
#     bbox_array_person_msg = None

# def callback_bbox_array_person(msg):
#     global bbox_array_person_msg
#     bbox_array_person_msg = msg

def callback_bbox_array_chair(msg):
    global bbox_array_chair_msg
    bbox_array_chair_msg = msg

def callback_bbox_array_person(msg):
    global bbox_array_chair_msg
    if bbox_array_chair_msg is not None:
        for i in range(len(bbox_array_chair_msg.boxes)):
            bbox_array_chair_msg.boxes[i].label = 56
            msg.boxes.append(bbox_array_chair_msg.boxes[i])
    pub.publish(msg)

if __name__ == '__main__':
    bbox_array_chair_msg = None
    rospy.init_node('relabel_bbox', anonymous=True)
    pub = rospy.Publisher("/bbox_array_relabel", BoundingBoxArray, queue_size=1)
    rospy.Subscriber("/bbox_array", BoundingBoxArray, callback_bbox_array_chair, queue_size=1)
    rospy.Subscriber("/bbox_array_center_person", BoundingBoxArray, callback_bbox_array_person, queue_size=1)
    rospy.spin()
