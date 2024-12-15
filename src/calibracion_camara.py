import cv2
from picamera2 import Picamera2


def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    for i in range(10):
        print("dale a cualquier tecla para tomar la foto")
        frame = picam.capture_array()
        cv2.imwrite(f"picam_1280_720_{i}.jpg", frame)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_video()
