import cv2
import numpy as np


def resize_image(img, fixed_width=300, use_ratio=False):
    """Resize image"""
    if use_ratio:
        ratio = fixed_width / img.shape[1]
        dim = (fixed_width, int(img.shape[0] * ratio))
    else:
        dim = (fixed_width, fixed_width)
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img_resized


def show_img(window_name, img, adjust=False):
    """3 arguments: window name, source images, boolean to adjust to screen size"""
    if adjust:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow(window_name)

    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rectanglesIntersect(r1, r2):
    """Calculates if 2 rectangles intersect"""
    x, y, xw, yh = r1
    x2, y2, xw2, yh2 = r2

    if x2 < xw and x < xw2 and y2 < yh:
        return y < yh2

    return False


def calculate_dist_height(
    intrinsic_params, extrinsic_params, real_height_signal, x, y, w, h
):
    """Calculate the distance to the signal"""
    K = intrinsic_params
    fx, fy = K[0, 0], K[1, 1]  # Longitudes focales en X e Y
    distance_z = (real_height_signal * fy) / h
    return distance_z


def contrast_enhance(img):
    """Enhance contrast with histogram equalization"""
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(img_lab)
    L = cv2.equalizeHist(L)
    img_lab_merge = cv2.merge((L, a, b))
    enhanced_img = cv2.cvtColor(img_lab_merge, cv2.COLOR_Lab2BGR)
    return enhanced_img


# canny edge detection
def auto_canny(img, method, sigma=0.33):
    """
    Apply Canny edge detection
    Args:
    img: grayscale image
    method: Otsu, triangle, and median
    sigma: 0.33 (default)
    2 outputs:
    edge_detection output, the high threshold for Hough Transform"""
    if method == "median":
        Th = np.median(img)

    elif method == "triangle":
        Th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE)

    elif method == "otsu":
        Th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    else:
        raise Exception("method specified not available!")

    lowTh = (1 - sigma) * Th
    highTh = (1 + sigma) * Th

    return cv2.Canny(img, lowTh, highTh), highTh


# Color based segmentation
# Color based segmentation (red, blue, yellow, black)
# Red color
lower_red1 = (0, 40, 50)
upper_red1 = (191, 250, 126)
lower_red1 = (0, 50, 50)
upper_red1 = (10, 255, 255)
lower_red2 = (170, 50, 50)
upper_red2 = (180, 255, 255)

# Blue color
lower_blue = (100, 40, 50)
upper_blue = (120, 255, 210)

# Yellow colors
lower_yellow = (20, 40, 50)
upper_yellow = (35, 255, 210)

# black colors
lower_black = (0, 0, 0)
upper_black = (179, 255, 5)


def color_seg(img, kernel_size=None):
    """
    Color based segmentation: red, blue
    Args:
    img: image in bgr
    kernel_size: None (default:(3, 3))"""
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)

    mask_combined = mask_red1 | mask_blue | mask_red2

    if kernel_size is not None:
        kernel = np.ones(kernel_size, np.uint8)
    else:
        kernel = np.ones((3, 3), np.uint8)

    # Apertura para limpiar
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)

    return mask_combined


# rectangle detection (using Douglas-Peuker algorithm)
def cnt_rect(cnts, coef=0.1):
    """Rectangle detection using Douglas-Peuker algorithm"""
    contour_list = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, coef * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 100:
            contour_list.append(cnt)
    if not contour_list:
        return None
    else:
        return sorted(contour_list, key=lambda x: -cv2.contourArea(x))


# circle detection
hough_circle_parameters = {
    "dp": 1,
    "minDist": 150,
    "param1": 200,  # adaptively change according to image
    "param2": 15,
    "minRadius": 1,
    "maxRadius": 500,
}


def cnt_circle(img, hough_dict):
    """
    Circle detection using Hough Circle Transform
    Args:
    img: Grayscale Image after resizing
    cnt: contour
    hough_dict: hough_circle_transform parameters"""
    mask = np.zeros_like(img)
    try:
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            hough_dict["dp"],
            hough_dict["minDist"],
            param1=hough_dict["param1"],
            param2=hough_dict["param2"],
            minRadius=hough_dict["minRadius"],
            maxRadius=hough_dict["maxRadius"],
        )
    except Exception as e:
        print("Error in Hough Circle Transform: ", e)
        print(
            "dp",
            hough_dict["dp"],
            "minDist",
            hough_dict["minDist"],
            "param1",
            hough_dict["param1"],
            "param2",
            hough_dict["param2"],
            "minRadius",
            hough_dict["minRadius"],
            "maxRadius",
            hough_dict["maxRadius"],
        )
        circles = None
    if circles is None:
        return circles
    else:
        # perform LCA
        list_circles = circles[0]
        largest_circles = max(list_circles, key=lambda x: x[2])
        center_x, center_y, r = largest_circles
        cv2.circle(mask, (int(center_x), int(center_y)), int(r), 255)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = cnts[0]
        return sorted(cnt, key=lambda x: -cv2.contourArea(x))


# combine the results of 2 shape detectors
def integrate_circle_rect(rect_cnt, circle_cnt, cnt):
    """Combine the results of 2 shape detectors"""
    if circle_cnt is not None and rect_cnt is not None:
        # compare the area
        if cv2.contourArea(circle_cnt) >= cv2.contourArea(rect_cnt):
            output = circle_cnt
        else:
            output = rect_cnt

    elif circle_cnt is not None and rect_cnt is None:
        output = circle_cnt

    elif circle_cnt is None and rect_cnt is not None:
        output = rect_cnt

    else:
        if len(cnt) == 0:
            return np.array([])
        else:
            output = max(cnt, key=cv2.contourArea)

    return output


# combine the results of edge detector + color based segmentation followed by shape detection combined results
def integrate_edge_color(output1, output2):
    """Combine the results of edge detector + color based segmentation followed by shape detection combined results"""
    if not isinstance(output1, np.ndarray):
        output1 = np.array(output1)

    if not isinstance(output2, np.ndarray):
        output2 = np.array(output2)

    if len(output1) == 0 and len(output2) == 0:
        return np.array([])

    elif len(output1) == 0 and output2.shape[-1] == 2:
        return output2

    elif len(output2) == 0 and output1.shape[-1] == 2:
        return output1

    else:
        if cv2.contourArea(output1[0]) > cv2.contourArea(output2[0]):
            return output1
        else:
            return output2


def mser(img):
    """Extract text regions using MSER"""
    # Create MSER object
    mser_object = cv2.MSER_create(delta=5, max_variation=0.4)

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_image = img.copy()

    # detect regions in gray scale image
    regions, _ = mser_object.detectRegions(gray)

    # Create convex hull of the regions
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    cv2.polylines(original_image, hulls, 1, (0, 255, 0))

    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    # Draw the contours
    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    # this is used to find only text regions, remaining are ignored
    resulting_detection = cv2.bitwise_and(img, img, mask=mask)

    return resulting_detection
