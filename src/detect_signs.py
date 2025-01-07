import glob
import cv2
from tqdm import tqdm
from tracking import track, show_img, resize_image, calculate_dist_height
from pathlib import Path
from classifying import classify_signal, draw_detection
from video import read_video, save_video, get_video_cap
import pickle

# Load the camera parameters
pickle_file = "calibration/camera_parameters.pkl"
# Abrir y cargar los datos del archivo .pickle
with open(pickle_file, "rb") as file:
    calibration_params = pickle.load(file)

# Guardar en variables separadas
intrinsics = calibration_params["intrinsics"]
extrinsics = calibration_params["extrinsics"]
h_señal = 0.5


def get_detection(img):
    """Get the detection of the signals in the image"""
    detection_image = img.copy()
    rectangles = track(img)

    results = {}
    for rect in rectangles:
        x, y, xw, yh = rect
        # Resize the signal to classify it
        signal_img = cv2.resize(
            detection_image[y:yh, x:xw], (500, 500), interpolation=cv2.INTER_AREA
        )
        signal_type = classify_signal(signal_img)
        distance = calculate_dist_height(
            intrinsics, extrinsics, h_señal, x, y, xw - x, yh - y
        )
        draw_detection(detection_image, rect, signal_type, distance)
        results[rect] = {"sign_type": signal_type, "distance": distance}

    return detection_image, results


def detect_signs_in_dir(img_dir, fixed_width=None) -> None:
    """Detects the signals in the images in the directory"""
    img_dir = Path(img_dir)

    for file in img_dir.glob("*.*"):
        str_fn = str(file)

        # Leer la imagen
        img = cv2.imread(str_fn)
        if img is None:
            print("No such file exists")
            break

        if fixed_width is not None:
            img = resize_image(img, fixed_width=fixed_width)

        detection_image, rectangles = get_detection(img)
        show_img("output", detection_image)


def detect_signs_in_video(video_path, video_name="result", fixed_width=None) -> None:
    """Detects the signals in the video and saves the result in a new video, does not use mean shift so it is very slow"""
    frames, frame_width, frame_height, frame_rate = read_video(video_path)
    results = []
    print("Analyzing frames...")
    for frame in tqdm(frames):
        if fixed_width is not None:
            frame = resize_image(frame, fixed_width=fixed_width)
        detection_image, rectangles = get_detection(frame)
        results.append(detection_image)

    frame_height, frame_width = results[0].shape[:2]
    save_video(video_name, results, frame_width, frame_height, frame_rate)


def get_tracking_windows_crop_hists(
    detection_image, rectangles
) -> tuple[list, list, list]:
    """Get the tracking windows, crop histograms and signal types"""
    tracking_windows = []
    crop_hists = []
    signal_types = []
    distances = []
    for x, y, xw, yh in rectangles.keys():
        crop = detection_image[y:yh, x:xw].copy()

        track_window = (x, y, xw - x, yh - y)

        # Convert the cropped object to HSV
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Compute the histogram of the cropped object (Reminder: Use only the Hue channel (0-180))
        mask = cv2.inRange(hsv_crop, (0, 60, 32), (180, 255, 255))

        crop_hist = cv2.calcHist(
            [hsv_crop], [0], mask=mask, histSize=[15], ranges=[0, 180]
        )
        cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)
        crop_hists.append(crop_hist)
        tracking_windows.append(track_window)
        signal_types.append(rectangles[(x, y, xw, yh)]["sign_type"])
        distances.append(rectangles[(x, y, xw, yh)]["distance"])

    return tracking_windows, crop_hists, signal_types, distances


def update_detected_signs(frame, results) -> tuple[list, list, list]:
    """Update the detected signs in the frame"""
    detection_image, rectangles = get_detection(frame)
    results.append(detection_image)

    tracking_windows, crop_hists, signal_types, distances = (
        get_tracking_windows_crop_hists(detection_image, rectangles)
    )
    return tracking_windows, crop_hists, signal_types, distances


def keep_track_of_detected_signs(
    frame, tracking_windows, crop_hists, signal_types, distances, results, term_crit
) -> None:
    """Keep track of the detected signs in the frame using mean shift"""
    input_frame = frame.copy()

    # Convert the frame to HSV
    img_hsv = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)

    for i, (crop_hist, track_window) in enumerate(zip(crop_hists, tracking_windows)):
        # Compute the back projection of the histogram
        img_bproject = cv2.calcBackProject([img_hsv], [0], crop_hist, [0, 180], 1)
        ret, track_window = cv2.meanShift(img_bproject, track_window, term_crit)
        x_, y_, w_, h_ = track_window
        # Compute the center of the object
        c_x = x_ + w_ // 2
        c_y = y_ + h_ // 2

        tracking_windows[i] = track_window

        cv2.circle(input_frame, (int(c_x), int(c_y)), 5, (0, 255, 0), -1)
        draw_detection(
            input_frame, (x_, y_, x_ + w_, y_ + h_), signal_types[i], distances[i]
        )
    results.append(input_frame)


def follow_signs_in_video(video_path, video_name="result", fixed_width=None):
    """Follows the detected signs in the video and saves the result in a new video, uses mean shift"""
    frames, frame_width, frame_height, frame_rate = read_video(video_path)
    results = []
    print("Analyzing frames...")

    update_frequency = 100

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)
    for frame_idx, frame in enumerate(tqdm(frames)):

        if fixed_width is not None:
            frame = resize_image(frame, fixed_width=fixed_width)

        if frame_idx % update_frequency == 0:
            tracking_windows, crop_hists, signal_types, distances = (
                update_detected_signs(frame, results)
            )
            continue

        keep_track_of_detected_signs(
            frame,
            tracking_windows,
            crop_hists,
            signal_types,
            distances,
            results,
            term_crit,
        )

    frame_height, frame_width = results[1].shape[:2]
    save_video(video_name, results, frame_width, frame_height, frame_rate)


def live_detection_video(video_path, fixed_width=None):
    """Detects the signals in the video and shows the result in a new window, this is done in real time"""
    cap, frame_width, frame_height, frame_rate = get_video_cap(video_path)
    update_frequency = 100
    i = 0
    results = []
    tracking_windows, crop_hists, signal_types = [], [], []
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if fixed_width is not None:
            frame = resize_image(frame, fixed_width=fixed_width)

        if i % update_frequency == 0:
            tracking_windows, crop_hists, signal_types, distances = (
                update_detected_signs(frame, results)
            )
        else:
            keep_track_of_detected_signs(
                frame,
                tracking_windows,
                crop_hists,
                signal_types,
                distances,
                results,
                term_crit,
            )
        cv2.imshow("Frame", results[-1])
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        i += 1
    cap.release()


if __name__ == "__main__":
    detect_signs_in_dir("../data/stop_sign/", fixed_width=500)
    # detect_signs_in_video("../data/video_prueba.mp4", "result_video_prueba")
    # for video in glob.glob("../data/videos/*.mp4"):
    #     follow_signs_in_video(
    #         video, video_name=video.split("\\")[-1].split(".")[0], fixed_width=None
    #     )
    # follow_signs_in_video(
    #     "../data/video_prueba_largo.mp4",
    #     "resize",
    #     fixed_width=500,
    # )
    # live_detection_video("../data/videos/video_80.mp4", fixed_width=None)
    # live_detection_video(0, fixed_width=None)
