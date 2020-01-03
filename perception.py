"""
main file for perception part of autonomous driving

todo:
- lane detection
- lane curvature calculation
- divergance from lane center calculation
- robust lane detection algorithm

backlog:
- image grabber from i.e. youtube
- parallelize lane and object detection
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageGrab, ImageFont, ImageDraw

"""
FUNCTIONS
"""

def make_coordinates(image, line_parameters):
    slope, intercept = (line_parameters)
    width = image.shape[1]
    y1 = image.shape[0]
    y2 = int(y1 *(3/5))
    x1 = int((y1 - intercept)//slope)
    x2 = int((y2 - intercept)//slope)
    #check if coordinates are within image dimension
    if x1 > width or x1 <= 0:
        #print("x1 = ", x1, " > ", width)
        print("coordinates",x1, x2, y1, y2, width)
    if x2 > width or x2 <= 0:
        #print("x2 = ", x2, " > ", width)
        print("coordinates",x1, x2, y1, y2, width)
    return np.array([x1, y1, x2, y2])

def make_coordinates2(image, start, end):
    # ms custom
    width = image.shape[1]
    y1 = image.shape[0]
    y2 = int(y1 *(3/5))
    return np.array([start, y1, end, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    max_slope = 0.5
    min_slope = 0.2
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # test = np.polyfit((x1,x2), (y1,y2), 2)
        # print(test)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0 and (abs(slope) >= min_slope and abs(slope) <= max_slope):
            left_fit.append((slope, intercept))
            # print("left slope: {}".format(slope))
        elif slope > 0 and (abs(slope) >= min_slope and abs(slope) <= max_slope):
            right_fit.append((slope, intercept))
            # print("right slope: {}".format(slope))
        else:
            print("slope is to steep: ", slope)
            continue
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
    else:
        left_line = None
    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
    else:
        right_line = None
    
    return np.array([left_line, right_line])

def calculate_average_slope(image, lines):
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    max_slope = 0.75
    min_y = image.shape[0] * (3/5)
    max_y = image.shape[0]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        if slope < 0 and abs(slope) < max_slope:
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        elif slope > 0 and abs(slope) < max_slope:
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])
        else:
            #print("slope is to steep: ", slope)
            continue

    left_fit_average = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
    left_x_start = int(left_fit_average(max_y))
    left_x_end = int(left_fit_average(min_y))
    right_fit_average = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
    rightt_x_start = int(right_fit_average(max_y))
    right_x_end = int(right_fit_average(min_y))

    left_line = make_coordinates2(image, left_x_start, left_x_end)
    right_line = make_coordinates2(image, right_x_start, right_x_end)
    return np.array([left_line, right_line])

def canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # detect big gradients between pixels and show
    canny = cv2.Canny(blur_image, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    horizontal_offset = width // 6
    vertical_offset = height // 4
    bottom_offset = 120
    flank = 50
    left_point_x = 0#(width*(1//8)) # 5 / 32
    bottom_points_y = height - bottom_offset
    right_point_x = width #(width*(7//8)) # 55 / 64
    center_point_x1 = (width//2) - horizontal_offset # 55 / 128
    center_point_x2 = (width//2) + horizontal_offset # 55 / 128
    center_point_y = (height//2) + 50
    p1 = (left_point_x, bottom_points_y)
    p2 = (width, bottom_points_y)
    p3 = ((right_point_x - flank), (height - vertical_offset))
    p4 = (center_point_x2, center_point_y)
    p5 = (center_point_x1, center_point_y)
    p6 = ((left_point_x + flank), (height - vertical_offset))
    polygons = np.array([
        [p1, p2, p3, p4, p5, p6]
    ])
    #[(200, height), (1100, height), (550, 250)]
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.int32(polygons), 255)
    masked_image = cv2.bitwise_and(image, mask) # 
    return masked_image

def print_shape(*args):
    output = ""
    for arg in args:
        #output = "{}image shape: ".format(arg)
        for dim in enumerate(arg.shape):
            #print(index, " , ", dim)
            output += str(dim[1])
            output += ","
        print(output, "\n eol \n")
        output = ""
    print("eof")


def get_HoughLinesP(image):
    pixel_resolution = 2
    radient_resolution = np.pi/180
    treshold = 75
    lines = cv2.HoughLinesP(image, pixel_resolution, radient_resolution, treshold, np.array([]), minLineLength=35, maxLineGap=5)
    return lines

def smooth_image(image,size):
    kernel = np.ones((size,size),np.float32)/(size**2)
    smooth_image = cv2.filter2D(image,-1,kernel)
    return smooth_image

def do_more_stuff(image):
    histogramm = np.sum(image[image.shape[0]//2,:,:], axis=0)
    midpoint = np.int(histogramm.shape[0]/2)
    leftx_base = np.argmax(histogramm[:midpoint])
    rightx_base = np.argmax(histogramm[:midpoint]) + midpoint
    # step 2 : create boxes around maxima
    # step 3 : stack and move boxes
    # step 4 : ployfit all points from boxes
    # left_fit = np.polyfit(lefty, leftx, 2)
    # right_fit = np.polyfit(righty, rightx, 2)

    # # calculate radius of curvature
    # # polyfit equation: y = Ax^2 + Bx + C
    # fit = [A, B, C]
    # curve_rad = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    # text_curvature = "radius of curvature: {}".format(curve_rad)
    # # distance to lane center
    # laneCenter = (rightPos - leftPos) / 2 + leftPos
    # distanceToCenter = laneCenter - imageCenter
    # if distanceToCenter > 0:
    #     print("right of center")
    #     leftright = "right"
    # else:
    #     print("left of center")
    #     leftright = "left"
    # text = "{}m {} of center".format(distanceToCenter, leftright)

    # # plotting line and drive area
    # polyimage = cv2.fillPoly(image, np.int_([pts]), (0, 255, 0))
    # polyimage = cv2.polylines(polyimage, np.int32([pts_left]), (0, 255, 0))
    #result = cv2.addWeighted(image, 1, polyimage, 0.5, 0)
    result = cv2.putTet(result, text_curvature, (50,100), font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
    result = cv2.putTet(result, text_distcenter, (50,100), font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
    return image

def show_results_in_one_image(one,two):
    # print("one: {} / two: {} / three: {} / four: {}".format(one.shape,two.shape,three.shape,four.shape))
    top_half = np.concatenate((one,two), axis=1)
    # bottom_half = np.concatenate((three,four), axis=1)
    # whole_img = np.concatenate((top_half,bottom_half), axis=0)
    return top_half

def image_preprocessing(image,smoothing):
    if smoothing:
        image = smooth_image(image, 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    image = canny(image)
    image = region_of_interest(image)
    return image

"""
CODE
"""
setting_smooth = False
setting_advanced = True
playback = 1
n_frame = 0

cap = cv2.VideoCapture("./videos/Dodge Charger SRT392 2015 on German Autobahn, 182 mph _ 292 kmh - totally legal.mp4")
while (cap.isOpened()):
    _, frame = cap.read()
    n_frame += 1
    print("frame: {}".format(n_frame))
    if frame is not None:
        # printscreen_numpy = cv2.cvtColor(frane, cv2.COLOR_BGR2RGB)
        preprocessed_image = image_preprocessing(frame, setting_smooth)

        # advanced stuff below
        if setting_advanced == False:
            cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            output = np.concatenate((frame_gray, preprocessed_image), axis=1)
            cv2.imshow("Result", output)
            # pass
        else:
            # try:
            lines = get_HoughLinesP(preprocessed_image)
            if lines is not None:
                averaged_lines = average_slope_intercept(frame, lines)
                # print("avgd lines: {}".format(averaged_lines))
                line_image = display_lines(region_of_interest(frame), averaged_lines)
                combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
                cv2.namedWindow('Res + Original', cv2.WINDOW_NORMAL)
                main_window = show_results_in_one_image(region_of_interest(frame), combo_image)
                cv2.imshow("Res + Original", main_window)
                cv2.namedWindow('Canny', cv2.WINDOW_NORMAL)
                cv2.imshow("Canny", preprocessed_image)
                # print("lines found")
            # except:
            #     # show your work
            #     lines = get_HoughLinesP(preprocessed_image)
            #     line_image = display_lines(frame, lines)
            #     cv2.namedWindow('Res + Original', cv2.WINDOW_NORMAL)
            #     main_window = show_results_in_one_image(frame, line_image)
            #     cv2.imshow("Res + Original", main_window)

    if cv2.waitKey(playback) == ord("q"):
        break
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     cv2.destroyAllWindows()
    #     break