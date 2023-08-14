''' 
    Purpose: Given same point clicked on bus lidar frame and the sensor node lidar frame, obtain transformation matrix for sensor node lidar frame with respect to local UTM frame (map)
    Subscribed topics: None
    Pulished topic: None

    Project: WATonoBus
    Author: Neel Bhatt
    Date: Feb 26, 2022
    Do not share, copy, or use without seeking permission from the author
'''

import numpy as np
from math import sin,cos,pi
from transformations import euler_from_matrix

print("\n-------------- FRAMES --------------\n")
print(" L -> Sensor Node LIDAR Frame\n B -> Bus LIDAR Frame\n G -> Ground Aligned Frame At Sensor Node LIDAR\n UTM -> Local UTM Frame (map)\n")

UTM_T_B = np.array([-0.8732913732528687, 0.4871982932090759, -5.162795846307254e-17, -395.07275390625, -0.4871982932090759, -0.8732913732528687, -1.653171143815163e-16, 601.6082763671875, -1.256284724305198e-16, -1.1921696241539679e-16, 1.0, 0.8600000143051147, 0.0, 0.0, 0.0, 1.0]).reshape(4,4)
UTM_R_B = UTM_T_B[:3,:3]
B_R_L = np.array([[ 3.51214273e-17,  1.00000000e+00,  5.01585965e-17],
                  [-5.73576436e-01,  6.12323400e-17, -8.19152044e-01],
                  [-8.19152044e-01,  0.00000000e+00,  5.73576436e-01]])

UTM_R_L = np.matmul(UTM_R_B,B_R_L)

L_p_La = np.array([9.29326057434,7.22492456436,-2.28016376495])
UTM_p_La = np.matmul(UTM_R_L,L_p_La)

UTM_p_UTMa = np.array([-398.52151929559955, 596.8680424120193, 3.319523692131041])
UTM_p_UTML = UTM_p_UTMa - UTM_p_La

print("Position on sensor node in LUTM:")
print(UTM_p_UTML)

UTM_T_L = np.vstack(( np.hstack(( UTM_R_L,UTM_p_UTML.reshape(-1,1) )) , np.array([0,0,0,1])  ))

print("Transformation Matrix UTM_T_L:")
print(UTM_T_L)


roll_alpha_L, pitch_beta_L, yaw_gamma_L = euler_from_matrix(UTM_T_L, 'sxyz')

print("-------------- RPY --------------\n", roll_alpha_L*180/pi, pitch_beta_L*180/pi, yaw_gamma_L*180/pi, "\n")

translation_x_L, translation_y_L, translation_z_L = UTM_T_L[:3,3]


print("\n-------------- Visualize TF in Rviz --------------\n")
print("rosrun tf static_transform_publisher "+str(translation_x_L)+" "+str(translation_y_L)+" "+str(translation_z_L)+" "+str(yaw_gamma_L)+" "+str(pitch_beta_L)+" "+str(roll_alpha_L)+" map rslidar_front 50")


transformation_matrix = UTM_T_L

np.save('transformation_matrix_UTM_T_L.npy', transformation_matrix)

# Results
# [-390.52508489  597.36415945   12.23996529]
# 43.47791019435533,-80.5456939582538