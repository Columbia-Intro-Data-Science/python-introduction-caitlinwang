import os
import cv2
import json
import time

import numpy as np

from math import floor
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

# -----------------------------------------------------------------------------
# Constant
# -----------------------------------------------------------------------------

ORIGINAL_IMAGES_PATH = "../original_images_v2/"
IMAGES_PATH = "../images_v2"
CLASSES = ["A", "B", "C", "D", "E",
           "F", "G", "H", "I", "K",
           "L", "M", "N", "O", "P",
           "Q", "R", "S", "T", "U",
           "V", "W", "X", "Y"]


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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Save processed image
def save_image(file_name, image):
    abs_path = os.path.abspath(IMAGES_PATH)

    if not os.path.exists(abs_path):
        os.mkdir(IMAGES_PATH)

    cv2.imencode(".jpg", image)[1].tofile(abs_path + "\\" + file_name)


# Load images
def load_images():
    result = []

    for index in range(len(CLASSES)):
        # Class name
        class_name = CLASSES[index]
        # Sub directory
        sub_dir = ORIGINAL_IMAGES_PATH + class_name
        # Image list in sub directory
        images = os.listdir(sub_dir)

        for image_name in images:
            # Image relative path
            image_path = os.path.join("%s/%s" % (sub_dir, image_name))
            # Read image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # Process image
            image_processed = apply_image_transformation(image)
            # Save image into folder
            save_image(class_name + "_" + image_name, image_processed)
            # Console log
            print("Load image " + class_name + "_" + image_name + " done")

            result.append({
                "image": image_processed.flatten().tolist(),
                "label": index,
                "class": class_name
            })

    return result


# Store json data
def store_json(data):
    with open("../output/dataset.json", "w") as json_file:
        json_file.write(json.dumps(data))


# Load json data
def load_json():
    with open("../output/dataset.json") as json_file:
        data = json.load(json_file)
        return data


# Format dataset & divide train, test
def format_dataset(dataset, ratio):
    images = []
    labels = []

    for item in dataset:
        images.append(np.array(item["image"]))
        labels.append(item["label"])

    return train_test_split(images, labels, test_size=ratio, random_state=42)


def print_with_precision(num):
    return "%0.5f" % num


# -----------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------

def generate_knn_classifier():
    num_neighbours = 10
    return KNeighborsClassifier(n_neighbors=num_neighbours)


def generate_logistic_classifier():
    return linear_model.LogisticRegression()


def generate_svm_classifier():
    return svm.LinearSVC()


def generate_classifier(model_name):
    classifier_generator_function_name = "generate_{}_classifier".format(model_name)
    return globals()[classifier_generator_function_name]()


# -----------------------------------------------------------------------------
# Run program
# -----------------------------------------------------------------------------

# Load images
dataset = load_images()

# Store dataset json
store_json(dataset)

# Load dataset json
dataset = load_json()

# Init dataset
training_images, testing_images, training_labels, testing_labels = format_dataset(load_json(), 0.2)
print("Format dataset & divide to train and test done")

beginTime = time.time()

# Generate model
classifier_model = generate_classifier("svm")
print("Generate model done")

# Train svm model
classifier_model = classifier_model.fit(training_images, training_labels)
joblib.dump(classifier_model, '../output/svm_model')
print("Train svm model done")

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))

# Test score
score = classifier_model.score(testing_images, testing_labels)
print("Model score: {}".format(print_with_precision(score)))

# Test predicted
# predicted = classifier_model.predict(testing_images)

# for i in range(len(predicted)):
#     print('Predicted: %s, Class: %s' % (CLASSES[predicted[i]], CLASSES[testing_labels[i]]))

# Train knn model
classifier_model = generate_classifier("knn")
classifier_model = classifier_model.fit(training_images, training_labels)
joblib.dump(classifier_model, '../output/knn_model')
print("Train knn model done")

# Test score
score = classifier_model.score(testing_images, testing_labels)
print("Model knn score: {}".format(print_with_precision(score)))

# Train logistic model
classifier_model = generate_classifier("logistic")
classifier_model = classifier_model.fit(training_images, training_labels)
joblib.dump(classifier_model, '../output/logistic_model')
print("Train logistic model done")

# Test score
score = classifier_model.score(testing_images, testing_labels)
print("Model logistic score: {}".format(print_with_precision(score)))
