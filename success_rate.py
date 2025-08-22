from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml 
from UI_inputs import get_user_input

# Get user input for paths and labels
project_folder, csv_folder, yaml_folder, labels = get_user_input()

csv_data_path = list(csv_folder.rglob("*.csv"))
baseline_data_path = yaml_folder
yaml_baseline_path = list(baseline_data_path.rglob("*.yaml"))  
postlesion_data_path = Path("//LaboDancauseDS/LabData/DANCN31/D/Machu/PostLesion/Task/")
yaml_postlesion_path = list(postlesion_data_path.rglob("*.yaml")) 

# --- Storage for success counters ---
success_trials_135_L = 0
failure_trials_135_L = 0
success_trials_135_R = 0
failure_trials_135_R = 0
success_trials_90_L = 0
failure_trials_90_L = 0
success_trials_90_R = 0
failure_trials_90_R = 0
success_trials_45_L = 0
failure_trials_45_L = 0
success_trials_45_R = 0
failure_trials_45_R = 0
success_trials_0_L = 0
failure_trials_0_L = 0
success_trials_0_R = 0
failure_trials_0_R = 0

# Processing YAML Files
# Function to extract hand and angle from YAML data
def get_hand_and_angle(yaml_file):
    with open(yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    hand = yaml_data.get('param', {}).get('main', 'Unknown')
    angle = yaml_data.get('param', {}).get('angles', 'Unknown')
    success = yaml_data.get('matlab_data', {}).get('grasp_offset', 'Unknown') > 0
    return hand, angle, success

# Loop through YAML files and process trial success/failure based on angle and hand
for idx, yaml_file in enumerate(yaml_baseline_path):
    hand, angle, success = get_hand_and_angle(yaml_file)

    # Handle angle and hand comparison
    if angle == [135.0, 135.0]:
        if hand == 'LEFT':
            if success:
                success_trials_135_L += 1
                print(f"Success trial for LEFT hand at 135°: {yaml_file.name}")
            else:
                failure_trials_135_L += 1
                print(f"Failure trial for LEFT hand at 135°: {yaml_file.name}")
        elif hand == 'RIGHT':
            if success:
                success_trials_135_R += 1
                print(f"Success trial for RIGHT hand at 135°: {yaml_file.name}")
            else:
                failure_trials_135_R += 1
                print(f"Failure trial for RIGHT hand at 135°: {yaml_file.name}")
    elif angle == [90.0, 90.0]:
        if hand == 'LEFT':
            if success:
                success_trials_90_L += 1
                print(f"Success trial for LEFT hand at 90°: {yaml_file.name}")
            else:
                failure_trials_90_L += 1
                print(f"Failure trial for LEFT hand at 90°: {yaml_file.name}")
        elif hand == 'RIGHT':
            if success:
                success_trials_90_R += 1
                print(f"Success trial for RIGHT hand at 90°: {yaml_file.name}")
            else:
                failure_trials_90_R += 1
                print(f"Failure trial for RIGHT hand at 90°: {yaml_file.name}")
    elif angle == [45.0, 45.0]:
        if hand == 'LEFT':
            if success:
                success_trials_45_L += 1
                print(f"Success trial for LEFT hand at 45°: {yaml_file.name}")
            else:
                failure_trials_45_L += 1
                print(f"Failure trial for LEFT hand at 45°: {yaml_file.name}")
        elif hand == 'RIGHT':
            if success:
                success_trials_45_R += 1
                print(f"Success trial for RIGHT hand at 45°: {yaml_file.name}")
            else:
                failure_trials_45_R += 1
                print(f"Failure trial for RIGHT hand at 45°: {yaml_file.name}")
    elif angle == [0.0, 0.0]:
        if hand == 'LEFT':
            if success:
                success_trials_0_L += 1
                print(f"Success trial for LEFT hand at 0°: {yaml_file.name}")
            else:
                failure_trials_0_L += 1
                print(f"Failure trial for LEFT hand at 0°: {yaml_file.name}")
        elif hand == 'RIGHT':
            if success:
                success_trials_0_R += 1
                print(f"Success trial for RIGHT hand at 0°: {yaml_file.name}")
            else:
                failure_trials_0_R += 1
                print(f"Failure trial for RIGHT hand at 0°: {yaml_file.name}")


# Total success and failure trials for each condition
success_L = [success_trials_135_L, success_trials_90_L, success_trials_45_L, success_trials_0_L]
failure_L = [failure_trials_135_L, failure_trials_90_L, failure_trials_45_L, failure_trials_0_L]
success_R = [success_trials_135_R, success_trials_90_R, success_trials_45_R, success_trials_0_R]
failure_R = [failure_trials_135_R, failure_trials_90_R, failure_trials_45_R, failure_trials_0_R]

# Plotting the success
angles = ['135°', '90°', '45°', '0°']

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(angles))

bar1 = ax.bar(index, success_L, bar_width, label='Success (Left)', color='b')
bar2 = ax.bar(index + bar_width, success_R, bar_width, label='Success (Right)', color='g')

ax.set_xlabel('Angles')
ax.set_ylabel('Number of Trials')
ax.set_title('Success Rate by Angle and Hand')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(angles)
ax.legend()

plt.tight_layout()
plt.show()

# Plotting the ratio success/failure
# Calculate the success to failure ratio for each condition (left and right hand)
# Avoid division by zero by ensuring failure count is non-zero
ratio_L = [s / f if f > 0 else np.nan for s, f in zip(success_L, failure_L)]
ratio_R = [s / f if f > 0 else np.nan for s, f in zip(success_R, failure_R)]

# Angles for labeling
angles = ['135°', '90°', '45°', '0°']

# Plotting the success/failure ratios
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(angles))

# Plot bars for left and right hand success/failure ratio
bar1 = ax.bar(index, ratio_L, bar_width, label='Success/Failure Ratio (Left)', color='b')
bar2 = ax.bar(index + bar_width, ratio_R, bar_width, label='Success/Failure Ratio (Right)', color='g')

# Customize plot
ax.set_xlabel('Angles')
ax.set_ylabel('Success/Failure Ratio')
ax.set_title('Success to Failure Ratio by Angle and Hand')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(angles)
ax.legend()

# Show plot
plt.tight_layout()
plt.show()
