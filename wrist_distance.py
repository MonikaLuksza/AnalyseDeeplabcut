from UI_inputs import get_user_input
import pandas as pd 
import numpy as np
import yaml
from pathlib import Path 

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

# Function to get the time of one movement from yaml files
def get_oneMvtTime_yaml(index1=0, index2=1): 
    heureMvt = 0

    for yaml_file in yaml_baseline_path[index1:index2]:
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)
            
        heureMvt = yaml_data.get('date').split('@')[1]
        totalMvtTime = float(yaml_data.get('matlab_data', {}).get('grasp_offset', 0)) #+ int(yaml_data.get('matlab_data', {}).get('delayReward', 0))

        if (yaml_data.get('matlab_data', {}).get('state') !=0):
            return heureMvt, float(totalMvtTime/1000)
        else :
            return heureMvt, None

# Function to see the distance between the wrist positions
def wrist_distance(index1, index2, csv_file):
    endDistanceX = 0
    beginningDistanceX = 0
    endDistanceY = 0
    beginningDistanceY = 0
    timeBeginningMvt = 0
    timeEndMvt = 0
    indexBeginningMvt = 0
    indexEndMvt = 0
    timeMvt = 0
    
    first_yaml_file = yaml_baseline_path[0]
    with open(first_yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)
        timeBeginningVideo = conversion_date_in_seconds(yaml_data.get('date').split('@')[1])

    for yaml_file in yaml_baseline_path[index1:index2]:
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)

    if (get_oneMvtTime_yaml(index1, index2)[1] is not None):
        timeBeginningMvt = conversion_date_in_seconds(get_oneMvtTime_yaml(index1, index2)[0])
        timeEndMvt = get_oneMvtTime_yaml(index1, index2)[1] + timeBeginningMvt
        timeMvt = timeEndMvt - timeBeginningMvt

    else:
        print("No valid movement time found in the YAML file.")

    # Read the CSV file
    position_data = pd.read_csv(csv_file, header=[0, 1, 2], low_memory=False)
    position_data.columns = [f"{bp}_{coord}" for _, bp, coord in position_data.columns.to_flat_index()]
    handUsed = yaml_data.get('param', {}).get('main', 'Unknown')
    if handUsed == 'LEFT':
        x_col = 'wrist1L_x'
        y_col = 'wrist1L_y'
        x2_col = 'wrist2L_x'
        y2_col = 'wrist2L_y'
    elif handUsed == 'RIGHT':
        x_col = 'wrist1R_x'
        y_col = 'wrist1R_y'
        x2_col = 'wrist2R_x'
        y2_col = 'wrist2R_y'

    position_data[x_col] = pd.to_numeric(position_data[x_col], errors='coerce')
    position_data[y_col] = pd.to_numeric(position_data[y_col], errors='coerce')
    X_point1 = position_data[x_col] 
    Y_point1 = position_data[y_col] 
    position_data[x2_col] = pd.to_numeric(position_data[x2_col], errors='coerce')
    position_data[y2_col] = pd.to_numeric(position_data[y2_col], errors='coerce')
    X_point2 = position_data[x2_col] 
    Y_point2 = position_data[y2_col] 

    indexBeginningMvt = int((timeBeginningMvt-timeBeginningVideo) * 10)
    print(f"Index of beginning of movement: {indexBeginningMvt}")
    indexEndMvt = int((timeMvt) * 10) + indexBeginningMvt
    print(f"Index of end of movement: {indexEndMvt}")

    num_points = len(position_data)
    for i in range(indexBeginningMvt, min(indexEndMvt+1, num_points)):
        print(f"Index: {i}")
        if i==indexBeginningMvt and i>0 and pd.notna(X_point1[i]) and pd.notna(Y_point1[i]) and pd.notna(X_point1[i - 1]) and pd.notna(Y_point1[i - 1]):
            beginningDistanceX = calculate_distance(X_point1[i], X_point2[i], 0, 0)
            beginningDistanceY = calculate_distance(0, 0, Y_point1[i], Y_point2[i])
            print(f"Wrist X distance at beginning: {beginningDistanceX}")
            print(f"Wrist Y distance at beginning: {beginningDistanceY}")
        if i==min(indexEndMvt, num_points-1) and i>0 and pd.notna(X_point1[i]) and pd.notna(Y_point1[i]) and pd.notna(X_point1[i - 1]) and pd.notna(Y_point1[i - 1]):
            endDistanceX = calculate_distance(X_point1[i], X_point2[i], 0, 0)
            endDistanceY = calculate_distance(0, 0, Y_point1[i], Y_point2[i])
            print(f"Wrist X distance at end: {endDistanceX}")
            print(f"Wrist Y distance at end: {endDistanceY}")

    # Sauvegarde dans resultats
    results.append({
        'Wrist X distance at the beginning': beginningDistanceX,
        'Wrist X distance at the end': endDistanceX,
        'Wrist Y distance at the beginning': beginningDistanceY,
        'Wrist Y distance at the end': endDistanceY
    })   

# Test of the trajectory length function 
#get_mvtTime_yaml(0, 100)
all_trials = []
for i in range(50):
    wrist_distance(i, i+1, csv_data_path[0])

# Sauvegarde des resultats dans fichier csv
results_df = pd.DataFrame(results)
output_csv_path = Path(project_folder) / "WRIST_DATA.csv"
results_df.to_csv(output_csv_path, index=False)
print(f"Saving...: {output_csv_path}")