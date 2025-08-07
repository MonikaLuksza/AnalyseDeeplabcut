import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# Load CSVs
cam1 = pd.read_csv("C:/Users/Usagers/2-Project1-Monika-2025-05-07/ANIPOSE/19-08/pose-3d/20240917-M-baseline71_2024-09-17_10-43-59_C1DLC_resnet50_Project1May7shuffle1_850000.csv", header=[0,1,2])
cam2 = pd.read_csv("C:/Users/Usagers/2-Project1-Monika-2025-05-07/ANIPOSE/19-09/pose-3D/20240917-M-baseline71_2024-09-17_10-43-59_C2DLC_resnet50_Project1May7shuffle1_850000.csv", header=[0,1,2])
cam3 = pd.read_csv("C:/Users/Usagers/2-Project1-Monika-2025-05-07/ANIPOSE/19-10/pose-3D/20240917-M-baseline71_2024-09-17_10-43-59_C3DLC_resnet50_Project1May7shuffle1_850000.csv", header=[0,1,2])

scorer1 = cam1.columns.levels[0][0]
scorer2 = cam2.columns.levels[0][0]
scorer3 = cam3.columns.levels[0][0]

# Target body parts
body_parts = ['JindexR', 'indexR', 'J3R']

# Frames to extract
frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]

# Initialize lists
pts_cam1_all = {bodypart: [] for bodypart in body_parts}
pts_cam2_all = {bodypart: [] for bodypart in body_parts}
pts_cam3_all = {bodypart: [] for bodypart in body_parts}

# Extract (x, y) for the desired body parts from each camera
for f in frames:
    for bodypart in body_parts:
        x1 = cam1[(scorer1, bodypart, 'x')][f]
        y1 = cam1[(scorer1, bodypart, 'y')][f]

        x2 = cam2[(scorer2, bodypart, 'x')][f]
        y2 = cam2[(scorer2, bodypart, 'y')][f]

        x3 = cam3[(scorer3, bodypart, 'x')][f]
        y3 = cam3[(scorer3, bodypart, 'y')][f]

        pts_cam1_all[bodypart].append([x1, y1])
        pts_cam2_all[bodypart].append([x2, y2])
        pts_cam3_all[bodypart].append([x3, y3])

# Convert to numpy arrays for each body part
for bodypart in body_parts:
    pts_cam1_all[bodypart] = np.array(pts_cam1_all[bodypart])
    pts_cam2_all[bodypart] = np.array(pts_cam2_all[bodypart])
    pts_cam3_all[bodypart] = np.array(pts_cam3_all[bodypart])

# Print the points for debugging
print("Points from Camera 1:\n", pts_cam1_all)
print("Points from Camera 2:\n", pts_cam2_all)
print("Points from Camera 3:\n", pts_cam3_all)

# Fake 3D world points (assuming wrist keypoints for calibration)
#X_world = np.array([[i, i*0.5, i*0.25] for i in range(len(pts_cam1_all['wrist1L']))]) 

X_world = np.array([
    [0.5, 0.0, 0.2],   # Point 1 (e.g., wrist position)
    [0.0, 0.0, 0.0],   # Point 2 (e.g., J2L position)
])

# Function to compute the Direct Linear Transform (DLT) for camera calibration, returns matrix A
def dlt(X_world, x_img):
    A = []
    for X, x in zip(X_world, x_img):
        X = np.array([*X, 1])  # Add homogeneous coordinate
        u, v = x
        row1 = np.hstack([X, np.zeros(4), -u * X])  # First row of the DLT matrix
        row2 = np.hstack([np.zeros(4), X, -v * X])  # Second row
        A.append(row1)
        A.append(row2)
    
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)  
    return P

# Triangulation function to calculate 3D points
def triangulate_point(P1, P2, P3, pt1, pt2, pt3):
    if np.any([pt1 == 0, pt2 == 0, pt3 == 0]):
        return np.nan
    
    pt1 = np.array(pt1).reshape(2, 1)  
    pt2 = np.array(pt2).reshape(2, 1)
    pt3 = np.array(pt3).reshape(2, 1)

    point_4d_1 = cv2.triangulatePoints(P1, P2, pt1, pt2)
    point_4d_2 = cv2.triangulatePoints(P1, P3, pt1, pt3)

    point_3d_1 = (point_4d_1 / point_4d_1[3])[:3].ravel()
    point_3d_2 = (point_4d_2 / point_4d_2[3])[:3].ravel()

    return np.mean([point_3d_1, point_3d_2], axis=0)

# Initialize dictionary to store 3D trajectories for each body part
trajectory_3d_all = {bodypart: [] for bodypart in body_parts}

# Process each body part
for bodypart in body_parts:
    pts_cam1 = pts_cam1_all[bodypart]
    pts_cam2 = pts_cam2_all[bodypart]
    pts_cam3 = pts_cam3_all[bodypart]

    P1 = dlt(X_world, pts_cam1)  # Camera 1 projection matrix
    P2 = dlt(X_world, pts_cam2)  # Camera 2 projection matrix
    P3 = dlt(X_world, pts_cam3)  # Camera 3 projection matrix

    trajectory_3d = []

    for f in range(len(pts_cam1)):
        pt1 = pts_cam1[f]  
        pt2 = pts_cam2[f]  
        pt3 = pts_cam3[f]  

        point_3d = triangulate_point(P1, P2, P3, pt1, pt2, pt3)
        trajectory_3d.append(point_3d)

    trajectory_3d_all[bodypart] = np.array(trajectory_3d)

# Print the 3D trajectory for each body part
for bodypart in body_parts:
    print(f"3D Trajectory for {bodypart}:\n", trajectory_3d_all[bodypart])

# Plot the 3D trajectories
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each body part's trajectory
for bodypart in body_parts:
    ax.plot(trajectory_3d_all[bodypart][:, 0], trajectory_3d_all[bodypart][:, 1], trajectory_3d_all[bodypart][:, 2], label=f'{bodypart} 3D Trajectory')


x_min, x_max = np.min([traj[:, 0] for traj in trajectory_3d_all.values()]), np.max([traj[:, 0] for traj in trajectory_3d_all.values()])
y_min, y_max = np.min([traj[:, 1] for traj in trajectory_3d_all.values()]), np.max([traj[:, 1] for traj in trajectory_3d_all.values()])
z_min, z_max = np.min([traj[:, 2] for traj in trajectory_3d_all.values()]), np.max([traj[:, 2] for traj in trajectory_3d_all.values()])

# Set equal scaling by setting limits for all axes to be the same range
axis_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
mid_x = (x_min + x_max) / 2
mid_y = (y_min + y_max) / 2
mid_z = (z_min + z_max) / 2

# Set the limits to ensure equal scaling on all axes
ax.set_xlim(mid_x - axis_range / 2, mid_x + axis_range / 2)
ax.set_ylim(mid_y - axis_range / 2, mid_y + axis_range / 2)
ax.set_zlim(mid_z - axis_range / 2, mid_z + axis_range / 2)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.legend()
plt.show()
