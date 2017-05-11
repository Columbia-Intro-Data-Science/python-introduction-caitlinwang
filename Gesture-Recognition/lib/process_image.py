import cv2
import numpy as np
from math import floor


# -----------------------------------------------------------------------------
# Images processing
# -----------------------------------------------------------------------------

def resize_image(frame, new_size):
    frame = cv2.resize(frame, (new_size, new_size))
    return frame


def make_background_black(frame):
    # Convert from RGB to HSV
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Prepare the first mask.
    # Tuned parameters to match the skin color of the input images...
    lower_boundary = np.array([0, 40, 30], dtype="uint8")
    upper_boundary = np.array([43, 255, 254], dtype="uint8")
    skin_mask = cv2.inRange(frame, lower_boundary, upper_boundary)

    # Apply a series of erosions and dilations to the mask using an
    # elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    # Prepare the second mask
    lower_boundary = np.array([170, 80, 30], dtype="uint8")
    upper_boundary = np.array([180, 255, 250], dtype="uint8")
    skin_mask2 = cv2.inRange(frame, lower_boundary, upper_boundary)

    # Combine the effect of both the masks to create the final frame.
    skin_mask = cv2.addWeighted(skin_mask, 0.5, skin_mask2, 0.5, 0.0)
    # Blur the mask to help remove noise.
    # skin_mask = cv2.medianBlur(skin_mask, 5)
    frame_skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
    frame = cv2.addWeighted(frame, 1.5, frame_skin, -0.5, 0)
    frame_skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

    return frame_skin


def make_skin_white(frame):
    height, width = frame.shape[:2]
    # Convert image from HSV to BGR format
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    # Convert image from BGR to gray format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Highlight the main object
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    threshold = 1
    for i in range(height):
        for j in range(width):
            if frame[i][j] > threshold:
                # Setting the skin tone to be white.
                frame[i][j] = 255
            else:
                # Setting everything else to be black.
                frame[i][j] = 0

    return frame


def remove_arm(frame):
    # Cropping 15 pixels from the bottom.
    height, width = frame.shape[:2]
    frame = frame[:height - 15, :]

    return frame


def find_largest_contour_index(contours):
    # if len(contours) <= 0:
    #     log_message = "The length of contour lists is non-positive!"
    #     raise Exception(log_message)

    largest_contour_index = 0

    contour_iterator = 1
    while contour_iterator < len(contours):
        if cv2.contourArea(contours[contour_iterator]) > cv2.contourArea(contours[largest_contour_index]):
            largest_contour_index = contour_iterator
        contour_iterator += 1

    return largest_contour_index


def draw_contours(frame):
    # "contours" is a list of contours found.
    _, contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Finding the contour with the greatest area.
    largest_contour_index = find_largest_contour_index(contours)

    # Draw the largest contour in the image.
    cv2.drawContours(frame, contours,
                     largest_contour_index, (255, 255, 255), thickness=-1)

    # Draw a rectangle around the contour perimeter
    contour_dimensions = cv2.boundingRect(contours[largest_contour_index])
    # cv2.rectangle(sign_image,(x,y),(x+w,y+h),(255,255,255),0,8)

    return frame, contour_dimensions


def centre_frame(frame, contour_dimensions):
    contour_perimeter_x, contour_perimeter_y, contour_perimeter_width, contour_perimeter_height = contour_dimensions
    square_side = max(contour_perimeter_x, contour_perimeter_height) - 1
    height_half = (contour_perimeter_y + contour_perimeter_y +
                   contour_perimeter_height) / 2
    width_half = (contour_perimeter_x + contour_perimeter_x +
                  contour_perimeter_width) / 2
    height_min, height_max = floor(height_half - square_side / 2), floor(height_half + square_side / 2)
    width_min, width_max = floor(width_half - square_side / 2), floor(width_half + square_side / 2)

    if 0 <= height_min < height_max and 0 <= width_min < width_max:
        frame = frame[height_min:height_max, width_min:width_max]
        # else:
        # log_message = "No contour found!!"
        # raise Exception(log_message)

    return frame


def apply_image_transformation(frame):
    frame = resize_image(frame, 100)
    frame = make_background_black(frame)
    frame = make_skin_white(frame)
    frame = remove_arm(frame)
    frame, contour_dimensions = draw_contours(frame)
    frame = centre_frame(frame, contour_dimensions)
    frame = resize_image(frame, 30)
    return frame
