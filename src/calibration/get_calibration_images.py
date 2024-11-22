import cv2
from picamera2 import Picamera2


def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    i = 0
    while True:
        i += 1
        input("Presiona enter para capturar imagen")
        frame = picam.capture_array()
        cv2.imwrite("picam_1280_720_{}.jpg".format(i), frame)
        cv2.imshow("picam", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    picam.stop()


if __name__ == "__main__":
    stream_video()
