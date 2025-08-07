import cv2
import numpy as np
import glob
import os
import json

# ========== CONFIG ==========
CHECKERBOARD = (8, 6)
SQUARE_SIZE = 1.8
camera_dirs = {
    "cam1": "C:/Users/Usagers/install-Monika/CALIBRATION/cam1",
    "cam2": "C:/Users/Usagers/install-Monika/CALIBRATION/cam2",
    "cam3": "C:/Users/Usagers/install-Monika/CALIBRATION/cam3",
}

# ========== PREPARE OBJECT POINTS ==========
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# ========== CALIBRATION STORAGE ==========
intrinsics = {}
distortions = {}
image_points = {}
object_points = {}
image_shape = {}
valid_filenames = {cam: [] for cam in camera_dirs}

# ========== STEP 1: INTRINSICS ==========
for cam_name, folder in camera_dirs.items():
    images = sorted(glob.glob(os.path.join(folder, "*.png")))
    objpoints, imgpoints = [], []

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"❌ Could not read image: {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        print(f"Processing {fname} for {cam_name}")
        print(f"Checkerboard found: {ret}")
        if ret:
            print(f"✅ Found corners in {fname} for {cam_name}")
            valid_filenames[cam_name].append(os.path.basename(fname))
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners2)
            image_shape[cam_name] = gray.shape[::-1]

    if len(objpoints) < 5:
        raise RuntimeError(f"Not enough valid frames for {cam_name}. Got {len(objpoints)}")

    ret, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_shape[cam_name], None, None)
    intrinsics[cam_name] = K
    distortions[cam_name] = dist
    object_points[cam_name] = objpoints
    image_points[cam_name] = imgpoints

# ========== STEP 2: EXTRINSICS ==========
def stereo_calibrate(camA, camB):
    common_fnames = sorted(set(valid_filenames[camA]).intersection(valid_filenames[camB]))
    if len(common_fnames) < 5:
        raise RuntimeError(f"Not enough common images between {camA} and {camB}. Got {len(common_fnames)}")

    camA_indices = [valid_filenames[camA].index(f) for f in common_fnames]
    camB_indices = [valid_filenames[camB].index(f) for f in common_fnames]

    obj_pts = [objp] * len(common_fnames)
    img_ptsA = [image_points[camA][i] for i in camA_indices]
    img_ptsB = [image_points[camB][i] for i in camB_indices]

    print(f"🔗 Calibrating {camA} and {camB} using {len(common_fnames)} common images...")

    flags = cv2.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        obj_pts, img_ptsA, img_ptsB,
        intrinsics[camA], distortions[camA],
        intrinsics[camB], distortions[camB],
        image_shape[camA],
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    )
    return R, T, ret

R31, T31, error_31 = stereo_calibrate("cam3", "cam1")
R32, T32, error_32 = stereo_calibrate("cam3", "cam2")

# ========== STEP 3: RODRIGUES ==========
def rodrigues_vec(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()

# ========== STEP 4: BUILD OUTPUT DICT ==========
camera_order = ["cam1", "cam2", "cam3"]
camera_names = ["A", "B", "C"]
extrinsics = {
    "cam1": (rodrigues_vec(R31), T31),
    "cam2": (rodrigues_vec(R32), T32),
    "cam3": (np.zeros(3), np.zeros(3)),  # reference
}

toml_data = {}
for idx, cam in enumerate(camera_order):
    rvec, tvec = extrinsics[cam]
    K = intrinsics[cam]
    dist = distortions[cam]
    size = image_shape[cam]

    cam_block = {
        "name": camera_names[idx],
        "size": [float(size[0]), float(size[1])],
        "matrix": [[float(x) for x in row] for row in K],
        "distortions": [float(x) for x in dist.flatten()],
        "rotation": [float(x) for x in rvec],
        "translation": [float(x) for x in tvec.flatten()],
    }
    toml_data[f"cam_{idx}"] = cam_block

toml_data["metadata"] = {
    "adjusted": False,
    "error": float((error_31 + error_32) / 2)
}

# ========== STEP 5: WRITE TO TOML ==========
def toml_full_precision(obj):
    lines = []
    for key in obj:
        lines.append(f"[{key}]")
        for k, v in obj[key].items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, list):
                if all(isinstance(i, (int, float)) for i in v):
                    lines.append(f"{k} = {json.dumps(v)}")
                elif all(isinstance(i, list) for i in v):
                    sub = ", ".join([json.dumps(i) for i in v])
                    lines.append(f"{k} = [{sub}]")
            elif isinstance(v, bool):
                lines.append(f"{k} = {str(v).lower()}")
            elif isinstance(v, float):
                lines.append(f"{k} = {repr(v)}")
            else:
                lines.append(f"{k} = {v}")
        lines.append("")
    return "\n".join(lines)

with open("calibration.toml", "w") as f:
    f.write(toml_full_precision(toml_data))

print("✅ calibration.toml written with full precision for Anipose")
