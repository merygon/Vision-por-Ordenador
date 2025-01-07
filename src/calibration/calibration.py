# %%
from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import random
import os


# %%
def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]


# %%
imgs_path = "../calibration_images/1280_720/*.jpg"
imgs_paths = glob.glob(imgs_path)

# %%
# Build a list containing the paths of all images from the left camera
imgs = load_images(imgs_paths)

# %%
# Find corners with cv2.findChessboardCorners()
corners = []
pattern_size = (9, 6)
for img in imgs:
    corners_found = cv2.findChessboardCorners(img, pattern_size, None)

    corners.append(corners_found)


# %%
corners_copy = copy.deepcopy(corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# To refine corner detections with cv2.cornerSubPix() you need to input grayscale images. Build a list containing grayscale images.
imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

corners_refined = [
    cv2.cornerSubPix(i, cor[1], pattern_size, (-1, -1), criteria) if cor[0] else []
    for i, cor in zip(imgs_gray, corners_copy)
]

# %%
imgs_copy = copy.deepcopy(imgs)

# %%
# Use cv2.drawChessboardCorners() to draw the cornes
corners_drawn = [
    cv2.drawChessboardCorners(img, pattern_size, np.array(corners), len(corners) > 0)
    for img, corners in zip(imgs_copy, corners_refined)
]

# %%
# Show images and save when needed


def show_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_image(img, filename):
    cv2.imwrite(filename, img)


os.makedirs("../calibration_images/1280_720_corners", exist_ok=True)
number_corners_detected = 0
for i, (img, corner_copy) in enumerate(zip(corners_drawn, corners_copy)):
    if corner_copy[0]:
        number_corners_detected += 1
        tag = "_detected"
    else:
        tag = ""
    write_image(
        img,
        f"../calibration_images/1280_720_corners/corners_{str(i).zfill(3)}{tag}.jpg",
    )
print(f"Number of corners detected: {number_corners_detected}/{len(corners_drawn)}")


# %%
# Design the method. It should return a np.array with np.float32 elements
def get_chessboard_points(chessboard_shape, dx, dy):
    nx = chessboard_shape[0]
    ny = chessboard_shape[1]
    N = nx * ny
    chessboard_points = np.zeros((N, 3), dtype=np.float32)
    # chessboard_points = []

    for y in range(ny):
        for x in range(nx):
            n = x + nx * y
            chessboard_points[n][0] = x * dx
            chessboard_points[n][1] = y * dy
            # chessboard_points.append((float(x * dx), float(y * dy), 0))
    return chessboard_points


chessboard_shape = pattern_size
dx = 34
dy = 34


# %%
# You need the points for every image, not just one
chessboard_points = [get_chessboard_points(pattern_size, dx, dy) for _ in imgs[1:]]
np.array(chessboard_points).shape

# %%
# Filter data and get only those with adequate detections
valid_corners = [cor[1] for cor in corners if cor[0]]
# Convert list to numpy array
valid_corners = np.asarray(valid_corners, dtype=np.float32)
valid_corners.shape

# %%
#
calibration_results_total = []
num_iter = 1
for _ in range(num_iter):
    random.shuffle(valid_corners)
    calibration_results_left = [
        cv2.calibrateCamera(
            chessboard_points[:i],
            valid_corners[:i],
            pattern_size,
            imgs_gray[0].shape[::-1],
            None,
            None,
        )
        for i in range(1, len(valid_corners) + 1)
    ]
    calibration_results_total.append(calibration_results_left)
calibration_results_total_rms = [
    [c[0] for c in random_shuffle] for random_shuffle in calibration_results_total
]
calibration_results_left = np.mean(calibration_results_total_rms, axis=0)
calibration_results_left_std = np.std(calibration_results_total_rms, axis=0) / np.sqrt(
    np.array(range(1, len(calibration_results_total_rms[0]) + 1))
)
rms, intrinsics, dist_coeffs, rvecs, tvecs = calibration_results_total[-1][-1]


# Obtain extrinsics
extrinsics = list(
    map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs)
)

# %%
# Print outputs
print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)
camera_parameters = {
    "intrinsics": intrinsics,
    "distortion_coefficients": dist_coeffs,
    "extrinsics": extrinsics,
}

# %%
#  Build a list containing the paths of all images from the fisheye camera and load images
fisheye_imgs_path = glob.glob(imgs_path)
fisheye_imgs = load_images(fisheye_imgs_path)


# %%
imgs_corners = []
# Parameters for cv2.cornerSubPix()
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

#  Complete the required parts of the loop
for img in fisheye_imgs:

    #  parse arguments to cv2.findChessboardCorners()
    corners = cv2.findChessboardCorners(img, pattern_size, None)
    if not corners[0]:
        continue
    #  convert image to grayscale to use cv2.cornerSubPix()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    refined_corners = cv2.cornerSubPix(
        gray_img, corners[1], (3, 3), (-1, -1), subpix_criteria
    )

    #  append only those refined_corners with proper detections
    imgs_corners.append(refined_corners)

# %%
#  Define the chessboard dimensions and the lenght of the squares (in [mm])
chessboard_dims = pattern_size
length = 34
fisheye_chessboard_points = [
    get_chessboard_points(chessboard_dims, length, length)[np.newaxis, ...]
    for _ in imgs_corners
]
# fisheye_chessboard_points.shape

# %%
# Parameters for cv2.fisheye.calibrate()
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
intrinsics = np.zeros((3, 3))
distortion = np.zeros((4, 1))
rotations = [np.zeros((1, 1, 3), dtype=np.float64) for _ in imgs_corners]
traslations = [np.zeros((1, 1, 3), dtype=np.float64) for _ in imgs_corners]


# %%
rms, _, _, _, _ = cv2.fisheye.calibrate(
    fisheye_chessboard_points,
    imgs_corners,
    gray_img.shape[::-1],
    intrinsics,
    distortion,
    rotations,
    traslations,
    calibration_flags,
    subpix_criteria,
)

# %%
# Show intrinsic matrix and distortion coefficients values
import pickle


print(intrinsics)
print(distortion)
distortion_parameters = {
    "intrinsics": intrinsics,
    "distortion_coefficients": distortion,
}
pickle.dump(camera_parameters, open("calibration/camera_parameters.pkl", "wb"))
pickle.dump(distortion_parameters, open("calibration/distortion_parameters.pkl", "wb"))
