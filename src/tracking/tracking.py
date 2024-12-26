from skimage.exposure import is_low_contrast
from pathlib import Path
import cv2
from utils.utils import *


def main_function():
    img_dir = "../../data/stop_sign/"
    img_dir = Path(img_dir)

    IOUs = []
    i = 0
    fixed_width = 500
    files_no_det = []
    img_iou_zero = []
    if list(img_dir.glob("*.*")) == []:
        print(
            "Directorio vacío de imagenes o no se está ejecutando desde el directorio adecuado. Muévete a src/tracking y vuelve a probar."
        )
    for file in img_dir.glob("*.*"):
        str_fn = str(file)
        img = cv2.imread(str_fn)
        if img is None:
            print("No such file exists")
            break
        img_copy = img.copy()
        i += 1
        filename = str_fn.split("\\")[-1]

        # Denoise the image + change to grayscale
        img_denoised = cv2.medianBlur(img_copy, 3)
        if is_low_contrast(img_denoised):
            img_denoised = contrast_enhance(img_denoised)
        img_denoised = contrast_enhance(img_denoised)
        # Resize the image
        ratio = fixed_width / img.shape[1]
        img_resized = cv2.resize(
            img_denoised, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA
        )
        img_resized_mser = mser(img_resized)

        # change to grayscale
        gray = cv2.cvtColor(img_resized_mser, cv2.COLOR_BGR2GRAY)

        # 1: Edge detection + shape detection + combine results of shape detector
        edge, canny_th2 = auto_canny(gray, "otsu")

        # Perform shape detectors
        cnts = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_edge = cnts[0]
        cnt = cnts[0]
        rect_edge = cnt_rect(cnt)
        # show_img("contours", cnts)
        hough_circle_parameters["param1"] = canny_th2
        circle_edge = cnt_circle(gray, hough_circle_parameters)

        output1 = integrate_circle_rect(rect_edge, circle_edge, cnt_edge)

        # color segmentation
        color_segmented = color_seg(img_resized_mser)

        # perform rectangular object detection
        cnts = cv2.findContours(color_segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = cnts[0]
        rect = cnt_rect(cnt)

        # perform circular object detection
        hough_circle_parameters["param1"] = 200
        circle = cnt_circle(color_segmented, hough_circle_parameters)

        output2 = integrate_circle_rect(rect, circle, cnt)

        # ground truth bb
        # gt_bb = np.array([annotations.loc[filename].x_start,
        #                   annotations.loc[filename].y_start,
        #                   annotations.loc[filename].x_end,
        #                   annotations.loc[filename].y_end])
        gt_bb = np.array([0, 2, 0, 2])
        gt_bb = (gt_bb * ratio).astype(int)

        cv2.rectangle(
            img_resized, (gt_bb[0], gt_bb[1]), (gt_bb[2], gt_bb[3]), (255, 0, 0), 2
        )

        # integrate output1 and output2
        final_output = integrate_edge_color(output1, output2)
        if len(final_output) == 0:
            print("no detection!")
            show_img("no detection", img_resized)
            IOUs.append(0)
            files_no_det.append(str_fn)
        else:
            x, y, w, h = cv2.boundingRect(final_output)
            cv2.rectangle(img_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

            pred_bb = (x, y, x + w, y + h)

            # IOU = computeIOU(gt_bb, pred_bb)

            # cv2.putText(img_resized_mser, f"IOU: {IOU:.3f}", (12, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
            # IOUs.append(IOU)

            # if IOU==0:
            #     img_iou_zero.append(img_resized_mser)

            # if i % 1000 == 0:
            #     show_img("results", img_resized_mser)
            show_img("results", img_resized)


if __name__ == "__main__":
    main_function()
