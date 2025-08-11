from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml 
from UI_inputs import get_user_input

# Results storage
results = []

# Get user input for paths and labels
project_folder, video_folder, csv_folder, yaml_folder, labels = get_user_input()

# Relevant paths
# Should I just load one video at a time?
video_path = list(video_folder.rglob("*.avi"))
csv_data_path = list(csv_folder.rglob("*.csv"))

# Relevant paths specific to Machu trials: pre and post lesion
# However, the yaml files chosen should be the ones that correspond to the csv files
baseline_data_path = yaml_folder
yaml_baseline_path = list(baseline_data_path.rglob("*.yaml"))  
postlesion_data_path = Path("//LaboDancauseDS/LabData/DANCN31/D/Machu/PostLesion/Task/")
yaml_postlesion_path = list(postlesion_data_path.rglob("*.yaml"))  

#Function to test the validity of the paths
def path_test():
# Test if paths are correct
    for csv_path in csv_data_path:
        print(csv_path)
        position_data = pd.read_csv(csv_path, header=0)
        print(position_data)
path_test()
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

#Function to get the angle between the index and the thumb
def calculate_angle(X_point1, X_point2, X_point3,
                    Y_point1, Y_point2, Y_point3,
                    Z_point1=None, Z_point2=None, Z_point3=None):  # Optional Z

    # 2D or 3D vectors depending on whether Z is provided
    if Z_point1 is not None and Z_point2 is not None and Z_point3 is not None:
        vectorA = np.array([X_point1 - X_point3, Y_point1 - Y_point3, Z_point1 - Z_point3])
        vectorB = np.array([X_point2 - X_point3, Y_point2 - Y_point3, Z_point2 - Z_point3])
    else:
        vectorA = np.array([X_point1 - X_point3, Y_point1 - Y_point3])
        vectorB = np.array([X_point2 - X_point3, Y_point2 - Y_point3])

    print(f"Vector A: {vectorA}")
    print(f"Vector B: {vectorB}")

    dot_product = np.dot(vectorA, vectorB)
    normA = np.linalg.norm(vectorA)
    normB = np.linalg.norm(vectorB)

    print(f"Dot product: {dot_product}")
    print(f"Norm A: {normA}")
    print(f"Norm B: {normB}")

    if normA != 0 and normB != 0:
        cos_theta = dot_product / (normA * normB)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle)
        return angle_degrees
    else:
        return 0  
  

# Function to get the movement time of multiple movements from yaml files
def get_mvtTime_yaml(index1=0, index2=100): 
    mvtNumber = 0
    heureMvt = 0

    for yaml_file in yaml_baseline_path[index1:index2]:
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)
        
        heureMvt = yaml_data.get('date').split('@')[1]
        totalMvtTime = float(yaml_data.get('matlab_data', {}).get('grasp_offset', 0)) 
        mvtNumber += 1
        if (yaml_data.get('matlab_data', {}).get('state') !=0):
            
            print(f"{heureMvt} Total {yaml_data.get('param', {}).get('main')} hand movement time (movement number: {mvtNumber}): {totalMvtTime} miliseconds")
        else :
            print(f"{heureMvt} Total {yaml_data.get('param', {}).get('main')} hand movement time (movement number: {mvtNumber}): ---")

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

#Function to calculate the trajectory length of one movement
def trajectory_one_movement(index1, index2, csv_file, cote) :  
    mvtDistance = 0
    distanceXaxis = 0
    distanceYaxis = 0
    ratioAxis = 0
    indexDebutMvt = 0
    indexFinMvt = 0
    tempsDebutMvt = 0
    tempsFinMvt = 0
    tempsMvt = 0
    emplacementFinalIndex = (0, 0)

    first_yaml_file = yaml_baseline_path[0]
    with open(first_yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)
        tempsDebutVideo = conversion_date_in_seconds(yaml_data.get('date').split('@')[1])
        print (f"Video start time: {tempsDebutVideo}")

    for yaml_file in yaml_baseline_path[index1:index2]:
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)

    if (get_oneMvtTime_yaml(index1, index2)[1] is not None):
        tempsDebutMvt = conversion_date_in_seconds(get_oneMvtTime_yaml(index1, index2)[0])
        print(f"Time of beginning of movement: {tempsDebutMvt}")
        tempsFinMvt = get_oneMvtTime_yaml(index1, index2)[1] + tempsDebutMvt
        print(f"Time of end of movement: {tempsFinMvt}")
        tempsMvt = tempsFinMvt - tempsDebutMvt
        print(f"Time of movement: {tempsMvt}")

        indexDebutMvt = int((tempsDebutMvt - tempsDebutVideo) * 10)
        print(f"Index of beginning of movement: {indexDebutMvt}")
        indexFinMvt = int((tempsMvt) * 10) + indexDebutMvt
        print(f"Index of end of movement: {indexFinMvt}")
    else:
        print("No valid movement time found in the YAML file.")
        

    # Read the CSV file
    position_data = pd.read_csv(csv_file, header=0, low_memory=False)
    if cote.upper() == 'L':
        x_col = 'JindexL_x'
        y_col = 'JindexL_y'
        z_col = 'JindexL_z' 
    elif cote.upper() == 'R':
        x_col = 'JindexR_x'
        y_col = 'JindexR_y'
        z_col = 'JindexR_z' 
    print(position_data.columns)
    print(f"Columns used for trajectory calculation: {x_col}, {y_col}, {z_col if z_col else 'None'}")
    print(position_data[x_col])
    position_data[x_col] = pd.to_numeric(position_data[x_col], errors='coerce')
    position_data[y_col] = pd.to_numeric(position_data[y_col], errors='coerce')
    X_point1 = position_data[x_col] 
    Y_point1 = position_data[y_col] 

    if z_col is not None:
        position_data[z_col] = pd.to_numeric(position_data[z_col], errors='coerce')
        Z_point1 = position_data[z_col]
    else:
        Z_point1 = None

    for i in range(indexDebutMvt, indexFinMvt + 1):
        if i > 0 and pd.notna(X_point1[i]) and pd.notna(Y_point1[i]) and pd.notna(X_point1[i - 1]) and pd.notna(Y_point1[i - 1]):
            x_prev, y_prev = X_point1[i - 1], Y_point1[i - 1]
            x_curr, y_curr = X_point1[i], Y_point1[i]
            if Z_point1 is not None and pd.notna(Z_point1[i]) and pd.notna(Z_point1[i - 1]):
                z_prev = Z_point1[i - 1]
                z_curr = Z_point1[i]
            else:
                z_prev = z_curr = None

            if all(v != 0 for v in [x_prev, y_prev, x_curr, y_curr]) and (z_prev is None or all(v != 0 for v in [z_prev, z_curr])):
                distance = calculate_distance(x_prev, x_curr, y_prev, y_curr, z_prev, z_curr)

                print(f"Distance between points: {distance}")
                mvtDistance += distance

                if i == indexFinMvt:
                    if z_curr is not None:
                        emplacementFinalIndex = (x_curr, y_curr, z_curr)
                    else:
                        emplacementFinalIndex = (x_curr, y_curr)

                distanceXaxis += abs(x_curr - x_prev)
                distanceYaxis += abs(y_curr - y_prev)
                print(f"Distance on X axis: {distanceXaxis}")
                print(f"Distance on Y axis: {distanceYaxis}")

                if distanceYaxis != 0:
                    ratioAxis = distanceXaxis / distanceYaxis
                    print(f"Ratio of X to Y axis: {ratioAxis}")

    print(f"Total movement distance: {mvtDistance}")

    # Sauvegarde dans resultats
    results.append({
        'csv_file': Path(csv_file).name,
        'movement_start_index': indexDebutMvt,
        'movement_end_index': indexFinMvt,
        'start_time_s': tempsDebutMvt,
        'end_time_s': tempsFinMvt,
        'duration_s': tempsMvt,
        'movement_distance': mvtDistance,
        'speed': mvtDistance / tempsMvt if tempsMvt > 0 else 0,
        'acceleration': mvtDistance / (tempsMvt ** 2) if tempsMvt > 0 else 0,
        'jerk': mvtDistance / (tempsMvt ** 3) if tempsMvt > 0 else 0,
        'X to Y axis movement ratio': ratioAxis,
        'side': cote.upper(),
        'hand reaching': yaml_data.get('param', {}).get('main', 'Unknown'),
        'angle of reach': yaml_data.get('param', {}).get('angles', 'Unknown'),
        'type of task': yaml_data.get('param', {}).get('tache', 'Unknown'),
        'index position at the end': emplacementFinalIndex
        })

# Function to calculate the grasp distance (between index and thumb)
def grasp_distance(index1, index2, csv_file):
    averageDistance = 0
    indexBeginningGrasp = 0
    indexEndGrasp = 0
    timeBeginningMvt = 0
    timeBeginningGrasp = 0
    timeEndMvt = 0
    timeEndGrasp = 0
    timeGrasp = 0
    mvtDistance = 0
    angle = 0
    
    first_yaml_file = yaml_baseline_path[0]
    with open(first_yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)
        timeBeginningVideo = conversion_date_in_seconds(yaml_data.get('date').split('@')[1])
        print (f"Video start time: {timeBeginningVideo}")

    for yaml_file in yaml_baseline_path[index1:index2]:
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)

    if (get_oneMvtTime_yaml(index1, index2)[1] is not None):
        timeBeginningMvt = conversion_date_in_seconds(get_oneMvtTime_yaml(index1, index2)[0])
        print(f"Time of the beginning of movement: {timeBeginningMvt}")
        timeBeginningGrasp = timeBeginningMvt + float((yaml_data.get('matlab_data', {}).get('cueOn', 'Unknown'))/1000)
        print(f"Time of the beginning of grasp: {timeBeginningGrasp}")
        timeEndMvt = get_oneMvtTime_yaml(index1, index2)[1] + timeBeginningMvt
        print(f"Time of the end of movement: {timeEndMvt}")
        timeEndGrasp = timeBeginningMvt + float((yaml_data.get('matlab_data', {}).get('grasp_onset', 'Unknown'))/1000)
        print(f"Time of the end of grasp: {timeEndGrasp}")
        timeMvt = timeEndMvt - timeBeginningMvt
        timeGrasp = timeEndGrasp - timeBeginningGrasp
        print(f"Time of movement: {timeMvt}")
        print (f"Time of grasp: {timeGrasp}")

        indexBeginningGrasp = int((timeBeginningGrasp - timeBeginningVideo) * 10)
        print(f"Index of beginning of grasp movement: {indexBeginningGrasp}")
        indexEndGrasp = int((timeGrasp) * 10) + indexBeginningGrasp
        print(f"Index of end of grasp movement: {indexEndGrasp}")
    else:
        print("No valid movement time found in the YAML file.")

      # Read the CSV file
    position_data = pd.read_csv(csv_file, header=0, low_memory=False)
    handUsed = yaml_data.get('param', {}).get('main', 'Unknown')
    if handUsed == 'LEFT':
        x_col = 'JindexL_x'
        y_col = 'JindexL_y'
        x2_col = 'JthumbL_x'
        y2_col = 'JthumbL_y'
        x3_col = 'wrist1L_x'
        y3_col = 'wrist1L_y'
        z_col = 'JindexL_z' 
        z2_col = 'JthumbL_z' 
        z3_col = 'wrist1L_z' 
        
    elif handUsed == 'RIGHT':
        x_col = 'JindexR_x'
        y_col = 'JindexR_y'
        x2_col = 'JthumbR_x'
        y2_col = 'JthumbR_y'
        x3_col = 'wrist1R_x'
        y3_col = 'wrist1R_y'
        z_col = 'JindexR_z' 
        z2_col = 'JthumbR_z' 
        z3_col = 'wrist1R_z' 

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

    if z_col is not None and z2_col is not None and z3_col is not None:
        position_data[z_col] = pd.to_numeric(position_data[z_col], errors='coerce')
        position_data[z2_col] = pd.to_numeric(position_data[z2_col], errors='coerce')
        position_data[z3_col] = pd.to_numeric(position_data[z3_col], errors='coerce')
        Z_point1 = position_data[z_col] 
        Z_point2 = position_data[z2_col] 
        Z_point3 = position_data[z3_col]
    else:
        Z_point1 = None
        Z_point2 = None
        Z_point3 = None

    for i in range(indexBeginningGrasp + 1, indexEndGrasp):
        if i > 0 and pd.notna(X_point1[i]) and pd.notna(Y_point1[i]) and pd.notna(X_point1[i - 1]) and pd.notna(Y_point1[i - 1]):
            z1 = Z_point1[i] if 'Z_point1' in locals() and Z_point1 is not None else None
            z2 = Z_point2[i] if 'Z_point2' in locals() and Z_point2 is not None else None
            z3 = Z_point3[i] if 'Z_point3' in locals() and Z_point3 is not None else None

            # Calculate grasping distance
            distance = calculate_distance(X_point1[i], X_point2[i], Y_point1[i], Y_point2[i], z1, z2)
            print(f"Grasping distance between points: {distance}")
            mvtDistance += distance
            averageDistance = mvtDistance / (indexEndGrasp - indexBeginningGrasp)        
            print(f"Total grasping movement distance: {mvtDistance}")
            print(f"Average grasping distance: {averageDistance}")

            # Calculate angle (extended to support Z if needed)
            angle = calculate_angle(
                X_point1[i], X_point2[i], X_point3[i],
                Y_point1[i], Y_point2[i], Y_point3[i],
                z1, z2, z3
            )
            print(f"Angle between index and thumb: {angle} degrees")

    # Sauvegarde dans resultats
    results.append({
        'Grasping distance': averageDistance,
        'Grasping angle': angle
    })

# Function to see the distance between the wrist positions
def wrist_distance(index1, index2, csv_file):
    endDistance = 0
    beginningDistance = 0
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
    position_data = pd.read_csv(csv_file, header=0, low_memory=False)
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
            beginningDistance = calculate_distance(X_point1[i], X_point2[i], Y_point1[i], Y_point2[i])
            print(f"Wrist distance at beginning: {beginningDistance}")
        if i==min(indexEndMvt, num_points-1) and i>0 and pd.notna(X_point1[i]) and pd.notna(Y_point1[i]) and pd.notna(X_point1[i - 1]) and pd.notna(Y_point1[i - 1]):
            endDistance = calculate_distance(X_point1[i], X_point2[i], Y_point1[i], Y_point2[i])
            print(f"Wrist distance at end: {endDistance}")

    # Sauvegarde dans resultats
    results.append({
        'Wrist distance at the beginning': beginningDistance,
        'Wrist distance at the end': endDistance
    })   




# Test of the trajectory length function 
#get_mvtTime_yaml(0, 100)
all_trials = []
for i in range(50):
    trajectory_one_movement(i, i+1, csv_data_path[0], 'L')  
    trajectory_one_movement(i, i+1, csv_data_path[0], 'R')
    grasp_distance(i, i+1, csv_data_path[0])
    wrist_distance(i, i+1, csv_data_path[0])

#For figure plotting
    #all_trials.append(speeds)

# for i, trial in enumerate(all_trials):
#    plt.plot(range(len(trial)), trial, label=f'Trial {i+1}')

#plt.xlabel('Index')
#plt.ylabel('Speed')
#plt.title('Per-Index Distance Across Multiple Trials')
#plt.legend()
#plt.grid(True)
#plt.show()

# Sauvegarde des resultats dans fichier csv
results_df = pd.DataFrame(results)
output_csv_path = Path(project_folder) / "TEST_CODE.csv"
results_df.to_csv(output_csv_path, index=False)
print(f"Saving...: {output_csv_path}")