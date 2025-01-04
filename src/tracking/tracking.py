import time
from skimage.exposure import is_low_contrast
from pathlib import Path
import cv2
from .utils import *


def track(img) -> set:

    img_copy = img.copy()

    # Denoise the image + change to grayscale
    img_denoised = cv2.medianBlur(img_copy, 3)
    if is_low_contrast(img_denoised):
        img_denoised = contrast_enhance(img_denoised)
    # img_denoised = contrast_enhance(img_denoised)
    # Resize the image
    img_resized = img_denoised  # resize_image(img_denoised, fixed_width)

    img_resized_mser = mser(img_resized)
    # change to grayscale
    gray = cv2.cvtColor(img_resized_mser, cv2.COLOR_BGR2GRAY)

    # 1: Edge detection + shape detection + combine results of shape detector
    edge, canny_th2 = auto_canny(gray, "otsu")
    # show_img("canny", edge)
    # Perform shape detectors
    cnts = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_edge = cnts[0]
    cnt = cnts[0]
    rect_edges = cnt_rect(cnt)
    # show_img("contours", cnts)
    hough_circle_parameters["param1"] = canny_th2 if canny_th2 > 0 else 100
    circle_edges = cnt_circle(gray, hough_circle_parameters)
    outputs1 = []
    if rect_edges is None and circle_edges is None:
        outputs1 = []
    elif rect_edges is None and circle_edges is not None:
        outputs1 = circle_edges
    elif rect_edges is not None and circle_edges is None:
        outputs1 = rect_edges
    else:
        for rect in rect_edges:
            for circle in circle_edges:
                # for rect, circle in zip(rect_edges, circle_edges):
                output1 = integrate_circle_rect(rect, circle, cnt_edge)
                outputs1.append(output1)

    # color segmentation
    color_segmented = color_seg(img_resized_mser)
    # show_img("segmented", color_segmented)
    # perform rectangular object detection
    cnts = cv2.findContours(color_segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0]
    rects = cnt_rect(cnt)

    # perform circular object detection
    hough_circle_parameters["param1"] = 100
    outputs2 = []
    circles = cnt_circle(color_segmented, hough_circle_parameters)
    if rects is None and circles is None:
        outputs2 = []
    elif rects is None and circles is not None:
        outputs2 = circles
    elif rects is not None and circles is None:
        outputs2 = rects
    else:
        for rect in rects:
            for circle in circles:
                # for rect, circle in zip(rects, circles):
                output2 = integrate_circle_rect(rect, circle, cnt)
                outputs2.append(output2)

    # integrate output1 and output2
    final_outputs = []
    for output1, output2 in zip(outputs1, outputs2):
        final_output = integrate_edge_color(output1, output2)
        final_outputs.append(final_output)

    rects_saved = list()
    # signs_detected = dict()
    for final_output in final_outputs:
        if len(final_output) == 0 or cv2.contourArea(final_output) <= 100:
            pass
        else:
            x, y, w, h = cv2.boundingRect(final_output)
            pred_bb = (x, y, x + w, y + h)

            # Check intersection with rects:
            intersect = False
            for rect_saved in rects_saved:
                if rect_saved == pred_bb or rectanglesIntersect(rect_saved, pred_bb):
                    intersect = True
                    break
            if not intersect:
                rects_saved.append(pred_bb)
            # prediction = predict_image(str_fn, bow.vocabulary, bow_extractor)
            # signs_detected[pred_bb] = prediction
    return rects_saved
