import numpy as np
import ros_numpy

import rospy

from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox

from typing import List

reduce = lambda x: x - np.min(x)

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
    points_np = np.zeros((points_np_record['x'].flatten().shape[0], 4))
    points_np[:, 0] = points_np_record['x'].flatten()
    points_np[:, 1] = points_np_record['y'].flatten()
    points_np[:, 2] = points_np_record['z'].flatten()
    points_np[:, 3] = points_np_record['intensity'].flatten()

    return points_np

def map_intensity(label: int, ids: list, max_val: int):
    label = label if label < 0 else ids[label]

    return label + 1
    # return 255.0 * (label / max_val + 1)

def make_bbox(frame: str, centroid: np.ndarray, id: int, pts: np.ndarray):
    bbox = BoundingBox()
    bbox.header.stamp = rospy.Time.now()
    bbox.header.frame_id = frame
    
    bbox.pose.position.x = centroid[0]
    bbox.pose.position.y = centroid[1]
    bbox.pose.position.z = centroid[2]
    
    bbox.dimensions.x = (np.max(pts[:, 0]) - centroid[0]) * 2
    bbox.dimensions.y = (np.max(pts[:, 1]) - centroid[1]) * 2
    bbox.dimensions.z = (np.max(pts[:, 2]) - centroid[2]) * 2

    bbox.label = id

    return bbox

def get_bbox(points_np: np.ndarray, labels: np.ndarray, centroids: List[np.ndarray],
             ids: List[int], counts: np.ndarray,
             stamp: rospy.Time, frame: str):
    
    msg = BoundingBoxArray()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame

    msg.boxes = [make_bbox(frame, centroids[i], ids[i], points_np[np.where(labels == i), : ][0])
                 for i in range(counts.shape[0] - 1)]
    
    return msg

def publish_points(points: List[np.ndarray]):
    points = np.array(points)
    msg = array_to_pointcloud2(points, rospy.Time.now(), 'ground_aligned')
    return msg

def hungarian_algorithm(cost_matrix: np.ndarray):
    num_agents, num_resources = cost_matrix.shape
    num_rows = max(num_agents, num_resources)
    num_cols = num_rows

    # Step 1: Input Preparation
    padded_matrix = np.zeros((num_rows, num_cols))
    padded_matrix[:num_agents, :num_resources] = cost_matrix
    cost_matrix = padded_matrix
    assignment_matrix = np.zeros_like(cost_matrix)

    # Step 2: Reduction
    cost_matrix = np.apply_along_axis(reduce, 1, cost_matrix) # Row reduction
    cost_matrix = np.apply_along_axis(reduce, 0, cost_matrix) # Column reduction

    # Step 3: Marking Zeros and Checking Assignment
    for _ in range(num_agents):
        marked_rows = np.zeros(num_rows, dtype=bool)
        marked_cols = np.zeros(num_cols, dtype=bool)
        assignment_found = False

        # Mark the first zero element in each row and column
        for i in range(num_rows):
            for j in range(num_cols):
                if cost_matrix[i, j] == 0 and not marked_rows[i] and not marked_cols[j]:
                    assignment_matrix[i, j] = 1
                    marked_rows[i] = True
                    marked_cols[j] = True
                    assignment_found = True
                    break

            if assignment_found:
                break

        if not assignment_found:
            break

        # Cover the marked zeros with a line
        for _ in range(num_agents):
            # Find an uncovered zero
            row, col = np.where(assignment_matrix == 1)
            uncovered_zero = np.where((~marked_rows[row]) & (~marked_cols[col]))[0]

            if len(uncovered_zero) == 0:
                break

            i, j = row[uncovered_zero[0]], col[uncovered_zero[0]]

            # If there is no starred zero in row i
            if not np.any(assignment_matrix[i] == 2):
                assignment_matrix[i, j] = 2

                # Find a starred zero in the same row (if exists)
                for star_col in np.where(assignment_matrix[i] == 1)[0]:
                    marked_rows[i] = True
                    marked_cols[star_col] = False
                    break

            # If there is a starred zero in row i
            else:
                for star_col in np.where(assignment_matrix[i] == 1)[0]:
                    marked_rows[i] = True
                    marked_cols[star_col] = True
                    break

    # Step 5: Assignment
    assignment_rows, assignment_cols = np.where(assignment_matrix == 1)
    assignment_pairs = list(zip(assignment_rows[:num_agents], assignment_cols[:num_resources]))

    # Step 6: Output
    return assignment_pairs
