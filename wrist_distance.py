from UI_inputs import get_user_input
import pandas as pd 
import numpy as np
import yaml
from pathlib import Path 
import math

# Results storage
results = []

# Get user input for paths and labels
project_folder, csv_folder, yaml_folder, labels = get_user_input()
csv_data_path = list(csv_folder.rglob("*.csv"))
baseline_data_path = yaml_folder
yaml_baseline_path = list(baseline_data_path.rglob("*.yaml"))  

#Function to convert the date from the yaml file into seconds
def conversion_date_in_seconds(date):
    date_str = str(date)
    h, m, s = map(int, date_str.split(':'))
    return h * 3600 + m * 60 + s

# Function to calculate the distance between two points
def calculate_distance(x1, x2, y1, y2, z1=None, z2=None):
    if z1 is not None and z2 is not None:
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    else:
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_rotation_angle(x1_start, y1_start, x2_start, y2_start,
                             x1_end, y1_end, x2_end, y2_end):
    # Create start and end vectors
    dx_start = x2_start - x1_start
    dy_start = y2_start - y1_start
    dx_end = x2_end - x1_end
    dy_end = y2_end - y1_end

    # Compute dot and cross products
    dot_product = dx_start * dx_end + dy_start * dy_end
    cross_product = dx_start * dy_end - dy_start * dx_end

    # Compute magnitudes
    magnitude_start = math.sqrt(dx_start**2 + dy_start**2)
    magnitude_end = math.sqrt(dx_end**2 + dy_end**2)

    # Avoid division by zero
    if magnitude_start == 0 or magnitude_end == 0:
        return None  

    # Clamp to avoid math domain error
    cos_theta = max(min(dot_product / (magnitude_start * magnitude_end), 1.0), -1.0)

    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    # Determine direction using cross product
    if cross_product < 0:
        angle_deg = -angle_deg  

    return angle_deg 

# Function to get the time of one movement from yaml files
def get_oneMvtTime_yaml(index1=0, index2=1): 
    heureMvt = 0

    for yaml_file in yaml_baseline_path[index1:index2]:
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)
            
        heureMvt = yaml_data.get('date').split('@')[1]
        totalMvtTime = float(yaml_data.get('matlab_data', {}).get('grasp_onset', 0)) 
        if (yaml_data.get('matlab_data', {}).get('state') !=0):
            return heureMvt, float(totalMvtTime/1000)
        else :
            return heureMvt, None

# Function to see the distance between the wrist positions
def wrist_distance(index1, index2, csv_file):
    endDistanceX = 0
    endDistance2X = 0
    beginningDistanceX = 0
    beginningDistance2X = 0
    endDistanceY = 0
    endDistance2Y = 0
    beginningDistanceY = 0
    beginningDistance2Y = 0
    angle_deg = 0
    angle_deg2 = 0
    timeBeginningMvt = 0
    timeEndMvt = 0
    indexBeginningMvt = 0
    indexEndMvt = 0
    timeMvt = 0
    indexBeginningReach = 0
    indexEndReach = 0
    
    first_yaml_file = yaml_baseline_path[0]
    with open(first_yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)
        timeBeginningVideo = conversion_date_in_seconds(yaml_data.get('date').split('@')[1])

    for yaml_file in yaml_baseline_path[index1:index2]:
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)

    if (get_oneMvtTime_yaml(index1, index2)[1] is not None):
        timeBeginningMvt = conversion_date_in_seconds(get_oneMvtTime_yaml(index1, index2)[0]) #+ 3.6626
        timeEndMvt = get_oneMvtTime_yaml(index1, index2)[1] + timeBeginningMvt
        timeMvt = timeEndMvt - timeBeginningMvt

    else:
        print("No valid movement time found in the YAML file.")

    # Read the CSV file. Uncomment if you want to use the xy coordinates from the top camera instead of the xyz coordinates of all three cameras
    #position_data = pd.read_csv(csv_file, header=[0, 1, 2], low_memory=False)
    position_data = pd.read_csv(csv_file, header=0, low_memory=False)
    #position_data.columns = [f"{bp}_{coord}" for _, bp, coord in position_data.columns.to_flat_index()]
    handUsed = yaml_data.get('param', {}).get('main', 'Unknown')
    if handUsed == 'LEFT':
        x_col = 'wrist1L_x'
        y_col = 'wrist1L_y'
        x2_col = 'wrist2L_x'
        y2_col = 'wrist2L_y'
        x3_col = 'indexL_x'
        y3_col = 'indexL_y'
        x4_col = 'f1L_x'
        y4_col = 'f1L_y'
    elif handUsed == 'RIGHT':
        x_col = 'wrist1R_x'
        y_col = 'wrist1R_y'
        x2_col = 'wrist2R_x'
        y2_col = 'wrist2R_y'
        x3_col = 'indexR_x'
        y3_col = 'indexR_y'
        x4_col = 'f1R_x'
        y4_col = 'f1R_y'

    position_data[x_col] = pd.to_numeric(position_data[x_col], errors='coerce')
    position_data[y_col] = pd.to_numeric(position_data[y_col], errors='coerce')
    X_point1 = position_data[x_col] 
    Y_point1 = position_data[y_col] 
    position_data[x2_col] = pd.to_numeric(position_data[x2_col], errors='coerce')
    position_data[y2_col] = pd.to_numeric(position_data[y2_col], errors='coerce')
    X_point2 = position_data[x2_col] 
    Y_point2 = position_data[y2_col] 
    position_data[x3_col] = pd.to_numeric(position_data[x3_col], errors='coerce')
    position_data[y3_col] = pd.to_numeric(position_data[y3_col], errors='coerce')
    X_point3 = position_data[x3_col]
    Y_point3 = position_data[y3_col]
    position_data[x4_col] = pd.to_numeric(position_data[x4_col], errors='coerce')
    position_data[y4_col] = pd.to_numeric(position_data[y4_col], errors='coerce')
    X_point4 = position_data[x4_col]
    Y_point4 = position_data[y4_col]

    # The time considered for the distances is the time of reach (exitHP until grasp_onset). Change lines timeBeginningReach and timeEndReach to change so. 
    if timeBeginningMvt != 0:
        indexBeginningMvt = int((timeBeginningMvt-timeBeginningVideo) * 10)
        print(f"Index of beginning of movement: {indexBeginningMvt}")
        timeBeginningReach = timeBeginningMvt + float((yaml_data.get('matlab_data', {}).get('exitHP', 'Unknown'))/1000)
        indexBeginningReach = int((timeBeginningReach - timeBeginningVideo) * 10)
        indexEndMvt = int((timeMvt) * 10) + indexBeginningMvt
        timeEndReach = timeBeginningMvt + float((yaml_data.get('matlab_data', {}).get('grasp_onset', 'Unknown'))/1000)
        indexEndReach = int((timeEndReach - timeBeginningVideo) * 10)
        print(f"Index of end of movement: {indexEndMvt}")

    num_points = len(position_data)
    for i in range(indexBeginningMvt, min(indexEndMvt+1, num_points)):
        print(f"Index: {i}")
        if i==indexBeginningReach and i>0 and pd.notna(X_point1[i]) and pd.notna(Y_point1[i]) and pd.notna(X_point1[i - 1]) and pd.notna(Y_point1[i - 1]):
            beginningDistanceX = calculate_distance(X_point1[i], X_point2[i], 0, 0)
            beginningDistanceY = calculate_distance(0, 0, Y_point1[i], Y_point2[i])
            print(f"Wrist X distance at beginning: {beginningDistanceX}")
            print(f"Wrist Y distance at beginning: {beginningDistanceY}")
            beginningDistance2X = calculate_distance(X_point3[i], X_point4[i], 0, 0)
            beginningDistance2Y = calculate_distance(0, 0, Y_point3[i], Y_point4[i])
        if i==min(indexEndReach, num_points-1) and i>0 and pd.notna(X_point1[i]) and pd.notna(Y_point1[i]) and pd.notna(X_point1[i - 1]) and pd.notna(Y_point1[i - 1]):
            endDistanceX = calculate_distance(X_point1[i], X_point2[i], 0, 0)
            endDistanceY = calculate_distance(0, 0, Y_point1[i], Y_point2[i])
            print(f"Wrist X distance at end: {endDistanceX}")
            print(f"Wrist Y distance at end: {endDistanceY}")
            endDistance2X = calculate_distance(X_point3[i], X_point4[i], 0, 0)
            endDistance2Y = calculate_distance(0, 0, Y_point3[i], Y_point4[i])

        # Calculate rotation angle using the wrist points
        if i > 0 and pd.notna(X_point1[i]) and pd.notna(Y_point1[i]):
            angle_deg = calculate_rotation_angle(
                X_point1[indexBeginningMvt], Y_point1[indexBeginningMvt],
                X_point2[indexBeginningMvt], Y_point2[indexBeginningMvt],
                X_point1[i], Y_point1[i],
                X_point2[i], Y_point2[i]
            )
            if angle_deg is not None:
                print(f"Rotation angle at index {i}: {angle_deg} degrees")
            else:
                print(f"Invalid vectors at index {i}, cannot compute angle.")

        # Calculate rotation angle using the knuckle points
        if i > 0 and pd.notna(X_point3[i]) and pd.notna(Y_point3[i]):
            angle_deg2 = calculate_rotation_angle(
                X_point3[indexBeginningMvt], Y_point3[indexBeginningMvt],
                X_point4[indexBeginningMvt], Y_point4[indexBeginningMvt],
                X_point3[i], Y_point3[i],
                X_point4[i], Y_point4[i]
            )
    # Sauvegarde dans resultats
    results.append({
        'movement_start_index': indexBeginningMvt,
        'movement_end_index': indexEndMvt,
        'start_time': timeBeginningMvt - timeBeginningVideo,
        'end_time': timeEndMvt - timeBeginningVideo,
        'Wrist X distance at the beginning': beginningDistanceX,
        'Wrist X distance at the end': endDistanceX,
        'Wrist Y distance at the beginning': beginningDistanceY,
        'Wrist Y distance at the end': endDistanceY,
        'Rotation angle using the wrist points': angle_deg,
        'Knuckle X distance at the beginning': beginningDistance2X,
        'Knuckle X distance at the end': endDistance2X,
        'Knuckle Y distance at the beginning': beginningDistance2Y,
        'Knuckle Y distance at the end': endDistance2Y,
        'Rotation angle using the knuckles points': angle_deg2,
        'Knuckles position at the beginning': (X_point1[indexBeginningMvt], Y_point1[indexBeginningMvt], X_point2[indexBeginningMvt], Y_point2[indexBeginningMvt]),
        'Hand reaching': yaml_data.get('param', {}).get('main', 'Unknown'),
        'angle of reach': yaml_data.get('param', {}).get('angles', 'Unknown')
    })   

# Results for first 100 trials
all_trials = []
for i in range(100):
    wrist_distance(i, i+1, csv_data_path[0])

# Sauvegarde des resultats dans fichier csv
results_df = pd.DataFrame(results)
output_csv_path = Path(project_folder) / "WRIST_DATA_xyz.csv"
results_df.to_csv(output_csv_path, index=False)
print(f"Saving...: {output_csv_path}")