import cv2
import os


def get_video_cap(videopath):
    cap = cv2.VideoCapture(videopath)

    #  Check if the video was successfully opened
    if not cap.isOpened():
        print("Error: Could not open the video file")
    # Get the szie of frames and the frame rate of the video
    frame_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )  # Get the width of the video frames
    frame_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )  # Get the height of the video frames
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video

    return cap, frame_width, frame_height, frame_rate


def read_video(videopath):
    """
    Reads a video file and returns its frames along with video properties.

    Args:
        videopath (str): The path to the video file.

    Returns:
        tuple: A tuple containing:
            - frames (list): A list of frames read from the video.
            - frame_width (int): The width of the video frames.
            - frame_height (int): The height of the video frames.
            - frame_rate (float): The frame rate of the video.
    """

    cap, frame_width, frame_height, frame_rate = get_video_cap(videopath)

    # Use a loop to read the frames of the video and store them in a list
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, frame_width, frame_height, frame_rate


def save_video(video_name, frames, frame_width, frame_height, frame_rate):
    # Create a folder to store the videos
    output_folder = "results/"
    folder_path = os.path.join("../", output_folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Name of the output video file with the parameters (history, varThreshold, detectShadows)
    videoname = f"{video_name}.avi"  # Name of the output video file with the parameters
    videopath = os.path.join(folder_path, videoname)

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec to use
    frame_size = (frame_width, frame_height)  # Size of the frames
    fps = frame_rate  # Frame rate of the video
    out = cv2.VideoWriter(videopath, fourcc, fps, frame_size)

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved at {videopath}")
