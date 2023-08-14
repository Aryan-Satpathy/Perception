''' 
    Purpose: Given roll, pitch, and yaw of ground aligned frame, obtain rotation matrix from sensor node lidar frame to ground aligned frame
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

print("\n-------------- LIDAR TO GROUND ALIGNED EXTRINSICS --------------\n")

# RPY angles in degrees
roll_alpha = 0.85
pitch_beta = -55
yaw_gamma  = 0

# Translation Vector: L_p_LG in meters
translation_x = 0
translation_y = 0
translation_z = 0


# ----------- Generate Rotation Matrix From G to L, given RPY angles (Fixed Axis: LIDAR) -----------
L_R_G = rot_matrix(roll_alpha,pitch_beta,yaw_gamma)
print("-------------- L_R_G --------------\n", L_R_G, "\n")

L_p_LG = np.array([[translation_x],[translation_y],[translation_z]])

rotation_matrix = L_R_G
translation_vector = L_p_LG

# ----------- Generate Transformation Matrix From G to L i,e. L_T_G, given Rotation and Translation -----------
L_T_G = np.hstack((rotation_matrix,translation_vector))
L_T_G = np.vstack((L_T_G,np.array([0,0,0,1])))
print("\n-------------- L_T_G --------------\n", L_T_G, "\n")

# ----------- Generate Transformation Matrix From L to G, given L_T_G -----------
G_T_L = np.linalg.inv(L_T_G)
print("\n-------------- G_T_L --------------\n", G_T_L, "\n")

transformation_matrix = G_T_L

# Test
print("--- Test ---")
point = np.array([1,0,1,1])
print("L_P: ",point)
print("A_P: ",G_T_L@point)

print("\n-------------- Visualize TF in Rviz --------------\n")
print("rosrun tf static_transform_publisher "+str(translation_x)+" "+str(translation_y)+" "+str(translation_z)+" "+str(yaw_gamma*pi/180)+" "+str(pitch_beta*pi/180)+" "+str(roll_alpha*pi/180)+" rslidar_front ground_aligned 50")

transformation_matrix = G_T_L

np.save('transformation_matrix_G_T_L.npy', transformation_matrix)