''' 
    Purpose: Given roll, pitch, and yaw of ground aligned frame, obtain rotation matrix from bus lidar frame to sensor node lidar
    Subscribed topics: None
    Pulished topic: None

    Project: WATonoBus
    Author: Neel Bhatt
    Date: Feb 26, 2022
    Do not share, copy, or use without seeking permission from the author
'''

# Run with Python3
import numpy as np
from math import sin,cos,pi
from transformations import euler_from_matrix #pip3 install transformations

def rot_matrix(roll,pitch,yaw):
	''' Compute Rotation Matrix from {B} to {A} = A_R_B given RPY angles using {A} as fixed axis about which RPY of {B} is given:
		Roll is about x axis, Pitch about y axis, and Yaw about z axis.
		
		Inputs: Roll, pitch, and yaw angles in degrees
		Outputs: A_R_B (3x3)
	'''

	alpha = yaw*pi/180; beta = pitch*pi/180; gamma = roll*pi/180
	Rz = np.array([[cos(alpha), -sin(alpha), 0],[sin(alpha),cos(alpha),0],[0,0,1]])
	Ry = np.array([[cos(beta), 0, sin(beta)],[0,1,0],[-sin(beta),0,cos(beta)]])
	Rx = np.array([[1,0,0],[0,cos(gamma),-sin(gamma)],[0,sin(gamma),cos(gamma)]])
	A_R_B = Rz@Ry@Rx
	# To check with final form of principal rotation matrix multiplication
	# R_XYZ = np.array([[cos(alpha)*cos(beta),cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma),cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma)],
	# 	[sin(alpha)*cos(beta),sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma),sin(alpha)*sin(beta)*cos(gamma)-cos(alpha)*sin(gamma)],
	# 	[-sin(beta),cos(beta)*sin(gamma),cos(beta)*cos(gamma)]])
	# print("\nR_XYZ:\n",R_XYZ)
	return A_R_B

print("\n-------------- FRAMES --------------\n")
print(" L -> Sensor Node LIDAR Frame\n B -> Bus LIDAR Frame\n G -> Ground Aligned Frame At Sensor Node LIDAR\n UTM -> Local UTM Frame (map)\n")

print("\n-------------- ROTATION MATRIX --------------\n")

# RPY angles in degrees
roll_alpha = 0
pitch_beta = 55
yaw_gamma  = -90

# ----------- Generate Rotation Matrix From L to B, given RPY angles (Fixed Axis: L) -----------
B_R_L = rot_matrix(roll_alpha,pitch_beta,yaw_gamma)
print("-------------- B_R_L --------------\n", B_R_L, "\n")

L_R_B = np.linalg.inv(B_R_L)

roll_alpha_B, pitch_beta_B, yaw_gamma_B = euler_from_matrix(np.insert(np.insert(L_R_B,3,[0.0],axis=0),3,[0.0,0.0,0.0,1.0],axis=1), 'sxyz')

print("-------------- RPY --------------\n", roll_alpha_B*180/pi, pitch_beta_B*180/pi, yaw_gamma_B*180/pi, "\n")

translation_x_B = 0
translation_y_B = 0
translation_z_B = 0


print("\n-------------- Visualize TF in Rviz --------------\n")
print("rosrun tf static_transform_publisher "+str(translation_x_B)+" "+str(translation_y_B)+" "+str(translation_z_B)+" "+str(yaw_gamma_B)+" "+str(pitch_beta_B)+" "+str(roll_alpha_B)+" rslidar_front bus_lidar 50")

rotation_matrix = L_R_B

np.save('rotation_matrix_L_R_B.npy', rotation_matrix)