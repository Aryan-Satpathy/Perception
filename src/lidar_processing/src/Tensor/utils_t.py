import numpy as np
import ros_numpy

import torch

import rospy

def array_to_pointcloud2(points, stamp=None, frame_id=None):
    all_dtype_names = ['x','y','z','intensity','ring','timestamp']
    all_dtype_formats = ['f4','f4','f4','u1','u2','f8']
    all_dtype_offsets = [0,4,8,16,18,24]
    all_dtype_itemsizes = [4,8,12,18,24,32]
    num_fields = points.shape[1]
    
    dtype_for_points = np.dtype(
                        {'names':all_dtype_names[:num_fields],
                        'formats':all_dtype_formats[:num_fields],
                        'offsets':all_dtype_offsets[:num_fields],
                        'itemsize':all_dtype_itemsizes[num_fields-1]})
    cloud_arr = np.rec.fromarrays(points.T, dtype=dtype_for_points)
    msg = ros_numpy.point_cloud2.array_to_pointcloud2(cloud_arr, stamp, frame_id)
    msg.header.stamp = rospy.Time.now()
    return msg

def pointcloud2_to_array(raw_cloud: ros_numpy.point_cloud2.PointCloud2) -> np.ndarray:
    points_np_record = ros_numpy.point_cloud2.pointcloud2_to_array(raw_cloud)  # np.array(...) allows for write access to points_np_record

    # Convert np record array to np array (with just x,y,z)
    points_np = torch.empty((points_np_record['x'].flatten().shape[0], 4))
    points_np[:, 0] = torch.from_numpy(points_np_record['x']).flatten()
    points_np[:, 1] = torch.from_numpy(points_np_record['y']).flatten()
    points_np[:, 2] = torch.from_numpy(points_np_record['z']).flatten()
    points_np[:, 3] = torch.from_numpy(points_np_record['intensity']).flatten()

    return points_np