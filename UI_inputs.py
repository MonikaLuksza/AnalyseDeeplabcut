# Code for the user interface to get input paths and labels for the project
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox

# These paths are default. If you chose to use the GUI, they will be overwritten.
# Change if you change the place of your project
config_path = Path("C:/Users/Usagers/Project1-Monika-2025-05-07") 
# Change if you change the place of your baseline data
baseline_data_path = Path("//LaboDancauseDS/LabData/DANCN31/D/Machu/Baseline/TaskData/")

def get_user_input():
    root = tk.Tk()
    root.withdraw()

    # Initial prompt
    messagebox.showinfo(title="Instructions", message="Select the folder of the project (1), the videos (2) and csv data (3) you would like to analyze. \nNext input the labels that were used in the videos (4).\n\nIf you not, default paths and labels will be used.")

    # Prompt for project folder
    project_folder = filedialog.askdirectory(title="Select Project Folder")
    if not project_folder:
        print("No folder selected for project: using default path")
        project_folder = config_path

    # Prompt for CSV folder
    csv_folder = filedialog.askdirectory(title="Select CSV Data Folder")
    if not csv_folder:
        print("No folder selected for CSV data: using default path")
        csv_folder = config_path / "labeled-data"
    else:
        csv_folder = Path(csv_folder)

    # Prompt for yaml folder
    yaml_folder = filedialog.askdirectory(title="Select yaml Data Folder")
    if not yaml_folder:
        print("No folder selected for yaml data: using default path")
        yaml_folder = baseline_data_path
    else:
        yaml_folder = Path(yaml_folder)

    # Prompt for labels
    labels = simpledialog.askstring("Input", "Enter labels separated by commas")
    if not labels:
        print("No labels provided: using default labels")
        labels = ['wrist1L', 'wrist1R', 'wrist2L', 'wrist2R', 
                  'J1L', 'J2L', 'J3L', 'JindexL', 'JthumbL', 'thumbL', 
                  'f1L', 'f2L', 'f3L', 'indexL', 
                  'J1R', 'J2R', 'J3R', 'JindexR', 'JthumbR', 
                  'thumbR', 'f1R', 'f2R', 'f3R', 'indexR']
    else:
        labels = [label.strip() for label in labels.split(',')]
    
    return Path(project_folder), Path(csv_folder), Path(yaml_folder), labels