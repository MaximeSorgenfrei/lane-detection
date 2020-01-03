import cv2
import numpy as np
import time

"""
TODO :
- image calibration for using own vides of driving scenes
- not 1 degree polyfit but polyfit of higher degree (capable of curves)
- histogramm of white (later yellow) points in image to "dectec" horizontal position of lanes in image
"""

def process_lines(image, lines):
    # empty list for line point seperation between left and right
    left_fit = []
    right_fit = []
    for line in lines:
        # for each line in lines: reshape point in line variable to coordinates, get slope and intercept of polyfit
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        # depending on slope distinguish between left and right
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # average out points and create coordinates for each side
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def make_coordinates(image, line_parameters):
    x1, y1, x2, y2 = None, None, None, None # default to prevent crashing when no lines found
    # try calculating coordinates from polynomial fit (1d)
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1*(3/5))
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
    except:
        # print("no valid lanes detected.")
        pass
    return np.array([x1, y1, x2, y2])

def draw_lines(img,lines):
    # draw all lines in <lines> to image, if lines existing else pass
    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4) # reorder variable to coordinates
            cv2.line(img, (x1, y1), (x2, y2), [255,0,0], 10)
    except:
        pass

def imp(image, pts1, pts2):
    # get perspective matrices from roi points in image to reduced image for processing
    M = cv2.getPerspectiveTransform(pts1,pts2)
    M_rev = cv2.getPerspectiveTransform(pts2,pts1)

    # warp, blur and apply filters to image
    dst = cv2.warpPerspective(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),M,(600,600))
    blur_image = cv2.GaussianBlur(dst, (5, 5), 0)
    canny_dst = cv2.Canny(blur_image, 50, 150)
    sobel_x = cv2.Sobel(blur_image, ddepth=-1, dx=1, dy=0, ksize=3)
    sobel_y = cv2.Sobel(blur_image, ddepth=-1, dy=1, dx=0, ksize=3)
    # empirically sobel_x and canny_dst worked best for used video
    best_of = cv2.bitwise_and(sobel_x, canny_dst)

    # find lines using HoughLinesP
    lines = cv2.HoughLinesP(best_of, 1, np.pi/180, 100, np.array([]), 10, 15)
    
    # process found lanes and if existing draw them to mask & best_of image
    mask = np.zeros_like(image)
    len_lines = 0
    try:
        processed_lines = process_lines(best_of, lines)
        len_lines = processed_lines.shape[0]
        draw_lines(mask, processed_lines)
        draw_lines(best_of, processed_lines)
    except:
        len_lines = 0
        pass

    # reverse warping of image to add lines to original rgb frame
    mask_transf = cv2.warpPerspective(mask, M_rev, (1280, 720))
    #  blend lines to original image
    result = cv2.addWeighted(image,1, mask_transf, 1, 0)

    return canny_dst, sobel_x, sobel_y, best_of, result, len_lines

def show_results_in_one_image(one, two, three, four):
    # put image pixelwise together:
    # one - top left, two - top right,  three - bottom left, four - bottom right
    # shape control
    # print("one: {} / two: {} / three: {} / four: {}".format(one.shape,two.shape,three.shape,four.shape))
    one = cv2.resize(one, dsize=(600,600))
    top_half = np.concatenate((one,two), axis=1)
    bottom_half = np.concatenate((three,four), axis=1)
    whole_img = np.concatenate((top_half,bottom_half), axis=0)
    return whole_img

def show_color_results(one, two):
    whole_img = np.concatenate((one,two), axis=1)
    return whole_img

# open video file and get amount of frames in file
cap = cv2.VideoCapture("./videos/highway45 내서 창녕.mp4")
total_frames = int(cap.get(7))
print("total frames: {}".format(total_frames))

# roi and warp transformations points
# pts1 = np.float32([[440,500],[800,500],[40,700],[1245,700]]) # old roi region
pts1 = np.float32([[0,720],[400,500],[800,500],[1280,720]])
pts2 = np.float32([[0,600],[0,0],[600,0],[600,600]])
polypoints = pts1.astype(np.int32)

# settings for time & performance tracking
start_time = time.time()
frames_with_lanes = 0
frames_with_two_lanes = 0
current_frame = 0
textColor=(0,0,0)

# loop of frames of video
while (cap.isOpened()):
    _, frame = cap.read()
    current_frame += 1
    # print("frame: {}".format(n_frame))
    if frame is not None:
        # convert image to grayscale to reduce computational complexity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # process grayscale image in imp-function
        canny, sobelx, sobely, best_of, result, num_lines = imp(frame, pts1, pts2)
        # aggregate images in one frame
        gray_window = show_results_in_one_image(gray, canny, sobelx, best_of)
        # main_window = show_color_results(frame, result)

        # performance statistics
        if num_lines > 0:
            frames_with_lanes += 1
        if num_lines == 2:
            frames_with_two_lanes += 1

        # written feedback in image
        if num_lines == 0: num_lines = "None"
        cv2.putText(result, "lanes found: {}".format(num_lines), (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, textColor, 3, cv2.LINE_AA)
        cv2.putText(result, "{:.2f}% / {:.2f}%".format((frames_with_lanes/current_frame)*100, (frames_with_two_lanes/current_frame)*100), (20, 60), cv2.FONT_HERSHEY_COMPLEX, 1, textColor, 3, cv2.LINE_AA)
        cv2.putText(result, "{:.0f}%".format((current_frame/total_frames)*100), (1220, 30), cv2.FONT_HERSHEY_COMPLEX, 1, textColor, 3, cv2.LINE_AA)
        print("loop took {:.4f} secs.".format(time.time() - start_time))
        
        # add ROI to rgb-frame
        result = cv2.addWeighted(result, 1, cv2.fillPoly(np.zeros_like(result), [polypoints], [0,255,0], cv2.LINE_AA), 0.2, 0)

        # reset time tracking for next frame
        start_time = time.time()

        # color
        cv2.namedWindow('Res + Original', cv2.WINDOW_NORMAL)
        cv2.imshow("Res + Original", result)
        # gray
        cv2.namedWindow('Res + Original (gray)', cv2.WINDOW_NORMAL)
        cv2.imshow("Res + Original (gray)", gray_window)
    if cv2.waitKey(1) == ord("q"):
        break