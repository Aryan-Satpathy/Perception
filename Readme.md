# A Brief Guide To Get Started With The Infrastrucutre Node Workspace

Hi Everyone,

Here are some utilities and code to help you.

## Startup

Run the following in a terminal after running `roscore`

`rosparam set use_sim_time true`

When you play the rosbags, please use the `--clock` flag as: `rosbag play 2022* --clock`

## Correcting Orientation of The Right Camera

Python run the rotated right image publisher node via `python ~/catkin_ws/src/Scripts/rotate_and_pub_right_image_compressed.py`.

## Visualizing The Data in Rviz

The `transform_publisher_infra` package includes a `Rviz` [folder](src/transform_publisher_infra/Rviz) with rviz configurations that you can directly use to visualize the cameras and LIDAR.

You may copy the `.rviz` files to `/home/YOUR_USERNAME/.rviz/` and open Rviz and File -> Open config to browse to this config file directly as well.

## Camera Intrinsic and Camera-LIDAR Extrinsic Calibration

The camera intrinsics are stored in `.yaml` files and can be found in [/src/transform_publisher_infra/transforms/Calibration/Intrinsics](/src/transform_publisher_infra/transforms/Calibration/Intrinsics).

The extrinsics for a given camera-LIDAR pair are stored in two `.npy` files per pair: 1. Transformation matrix (`4x4`) and 2. Projection Matrix (`3x4`) and can be found in [/src/transform_publisher_infra/transforms/Calibration/Extrinsics](/src/transform_publisher_infra/transforms/Calibration/Extrinsics).

The results for the LIDAR points projection onto the left and right images are shown below:

### Projection Results - Left Camera

<img src="images/calibration/left/1.png" alt="left" width="500"/>
<img src="images/calibration/left/2.png" alt="left" width="500"/>
<img src="images/calibration/left/3.png" alt="left" width="500"/>
<img src="images/calibration/left/4.png" alt="left" width="500"/>
<img src="images/calibration/left/5.png" alt="left" width="500"/>

### Projection Results - Right Camera

<img src="images/calibration/right/1.png" alt="right" width="500"/>
<img src="images/calibration/right/2.png" alt="right" width="500"/>
<img src="images/calibration/right/3.png" alt="right" width="500"/>
<img src="images/calibration/right/4.png" alt="right" width="500"/>
<img src="images/calibration/right/5.png" alt="right" width="500"/>

## Transforming and Projecting Points Between Frames:

Let frame **{LC}** correspond to the left camera and **{RC}** correspond to the right camera.

To obtain the 3D position of a point `p = [x, y, z, 1]^T` expressed in the LIDAR frame **{L}** in the left camera camera frame **{LC}**, for instance, you can obtain the transformation matrix `LC_T_L` from the [numpy file](/src/transform_publisher_infra/transforms/Calibration/Extrinsics/Left_Infra/LC_T_L_final.npy) `LC_T_L_final.npy` and perform `LC_T_L @ p`.

where @ is the matrix multiplication operator resulting in `p` expressed in **{LC}**. This is the 3D point expressed in **{LC}**. However, if you wanted to project point `p`into the camera frame to obtain `[u, v, 1]`then: you need to perform `LC_proj_L @ p` and divide the resulting 3 elements by the third element to rescale the third element to 1. You can obtain `LC_proj_L` from the [numpy file](/src/transform_publisher_infra/transforms/Calibration/Extrinsics/Left_Infra/LC_proj_L_final.npy) `LC_proj_L_final.npy`.


## Frame TF Publisher and Ground Plane Filtering


### Transform Publisher

I have prepared a package named `transform_publisher_infra` that publishes transforms between the following frames:

1.  `rslidar_front` - Sensor Node LIDAR frame - denoted by **{L}**
2.  `ground_aligned` - Frame aligned with the ground plane of the test setting in X Lot - denoted by **{G}**
3.  `map` - The local map frame (The map frame is in front of E3 intersection at `easting = 537132` and `northing = 4813391` (UTM)) - denoted by **{UTM}**
4.  `bus_lidar` - This is the bus lidar frame when it was nearly aligned with the sensor node in the for_calib bags - denoted by **{B}**

Here is an illustration of some of the frames:

<img src="images/frames/camera_frames.png" alt="right" width="500"/>
<img src="images/frames/ground_aligned_vs_rslidar_front.png" alt="right" width="500"/>

Other information:

-   The package includes a `transforms` folder with transformation and rotation matrices between these frames as well
-   The package includes a `Rviz` folder with rviz configurations that you can directly use

I have attached a screenshot confirming that the map transform works well.

Launch using: `roslaunch transform_publisher_infra transform_pub.launch`

### Ground Filtering

Upon request and to help save your time for 2D, 3D detection, and global fusion, I have included two scripts that enable ground filtering of the pointcloud:

1.  `transform_points_infra.cpp` - This script transforms the raw lidar points from frame **{L}** to **{G}** - see file header for details on topics
2.  `ground_filter.py` - This script filters ground points based on vectorized thresholding that take 3ms on average and is efficient t o use.

Here are some screenshots of the performance:

<img src="images/ground_plane_filtering/1.png" alt="right" width="800"/>


<img src="images/ground_plane_filtering/2.png" alt="right" width="800"/>


<img src="images/ground_plane_filtering/3.png" alt="right" width="800"/>


Launch 1. using: `rosrun transform_publisher_infra transform_points_infra`

Launch 2. by moving to the directory of script and using: `python ~/catkin_ws/src/Scripts/ground_filter.py`

### Comparison of Bus GPS Position With Sensor Node Detections

To obtain the bus location in **{UTM}** frame you will need to subscribe to the bus topic `/ego_odom` which contains the map-based location of the bus in easting and northing. You can use the `pose.position.x` field for easting and `pose.position.y` field for northing from `/ego_odom`.

To obtain the map-based location of a point `p = [x, y, z, 1]^T` expressed in the sensor node frame **{L}**, you can obtain the transformation matrix `UTM_T_L` from the [numpy file](/src/transform_publisher_infra/transforms/transformation_matrix_UTM_T_L.npy) `transformation_matrix_UTM_T_L.npy` and perform `UTM_T_L @ p`

where @ is the matrix multiplication operator resulting in `p` expressed in **{UTM}**.

Thanks,

Neel
