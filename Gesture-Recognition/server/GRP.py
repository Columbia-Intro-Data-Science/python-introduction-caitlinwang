import os
import base64
import cv2
import io
import sys
import lib
import numpy as np

sys.path.append(os.path.abspath('/Users/nanbinx/finalv3/Gesture-Recognition/'))
# print(sys.path)

from sklearn.externals import joblib
from PIL import Image

# Flask
from flask import Flask
from flask import request
from flask import jsonify
from flask import abort
from flask import render_template
from flask import redirect
from flask import url_for

# -----------------------------------------------------------------------------
# Constant
# -----------------------------------------------------------------------------

ORIGINAL_IMAGES_PATH = "/Users/nanbinx/finalv3/Gesture-Recognition/original_images_v2/"
IMAGES_PATH = "/Users/nanbinx/finalv3/Gesture-Recognition/images_v2"
CLASSES = ["A", "B", "C", "D", "E",
           "F", "G", "H", "I", "K",
           "L", "M", "N", "O", "P",
           "Q", "R", "S", "T", "U",
           "V", "W", "X", "Y"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def init_model():
    return joblib.load('/Users/nanbinx/finalv3/Gesture-Recognition/output/svm_model')


def process_image(image_base64):
    # Split base64 code
    encoded_data = image_base64.split(',')[1]
    # Decode
    image = base64.b64decode(encoded_data)
    # Nplise
    image_np = np.fromstring(image, dtype=np.uint8)
    # Read image with cv2
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # Processing
    image_processed = lib.process_image.apply_image_transformation(frame)

    return image_processed


# List all processed images base64 code
def list_processed_images_src():
    result = []

    # Processed image list in directory
    images = os.listdir(IMAGES_PATH)

    for image_name in images:
        # Image relative path
        image_path = os.path.abspath(os.path.join("%s/%s" % (IMAGES_PATH, image_name)))
        # Read image
        image_base64 = base64.b64encode(open(image_path, 'rb').read()).decode("utf-8")
        # Console log
        print("Load processed image src " + image_name + " done")

        result.append(image_base64)

    return result


# List all original images base64 code
def list_original_images_src():
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
            image_path = os.path.abspath(os.path.join("%s/%s" % (sub_dir, image_name)))
            # Read image
            image_base64 = base64.b64encode(open(image_path, 'rb').read()).decode("utf-8")
            # Console log
            print("Load original image src " + image_name + " done")

            result.append(image_base64)

    return result


# List original images base64 code by class name
def list_original_images_by_class(class_name):
    result = []

    # Sub directory
    sub_dir = ORIGINAL_IMAGES_PATH + class_name
    # Image list in sub directory
    images = os.listdir(sub_dir)

    for image_name in images:
        # Image relative path
        image_path = os.path.abspath(os.path.join("%s/%s" % (sub_dir, image_name)))
        # Read image
        image_base64 = base64.b64encode(open(image_path, 'rb').read()).decode("utf-8")
        # Console log

        result.append(image_base64)

    return result


# -----------------------------------------------------------------------------
# Flask
# -----------------------------------------------------------------------------

# Init Flask application
app = Flask(__name__)

# Init classifier model
classifier_model = init_model()

# Original images
original_images = list_original_images_src()

# Processed images
processed_images = list_processed_images_src()


# -----------------------------------------------------------------------------
# Http
# -----------------------------------------------------------------------------

@app.route('/')
def hello_world():
    return redirect(url_for('render_original_images'))


@app.route('/classifier')
def classifier():
    return render_template('classifier.html', title='Classifier')


@app.route('/original_images')
def render_original_images():
    # return render_template('original_images.html', title='Original Images', images=original_images)
    return render_template('original_images.html', title='Original Images')


@app.route('/processed_images')
def render_processed_images():
    return render_template('processed_images.html', title='Processed Images', images=processed_images)


@app.route('/api/image/list/original', methods=['GET'])
def list_original_images():
    class_name = request.args.get('c')
    return jsonify({'status': 0, 'data': list_original_images_by_class(class_name)})


@app.route('/api/image/processing', methods=['POST'])
def processing_image():
    if not request.json:
        abort(400)

    image_base64 = request.json.get('image')
    image_processed = process_image(image_base64)

    # Convert np image to base64
    image_pil = Image.fromarray(image_processed)
    buff = io.BytesIO()
    image_pil.save(buff, format="JPEG")
    image_processed = base64.b64encode(buff.getvalue()).decode("utf-8")

    return jsonify({'status': 0, 'data': image_processed})


@app.route('/api/image/predict', methods=['POST'])
def predict():
    if not request.json:
        abort(400)

    image_base64 = request.json.get('image')
    image_flattened = process_image(image_base64).flatten().reshape(1, -1)
    labels = classifier_model.predict(image_flattened)

    return jsonify({'status': 0, 'data': CLASSES[labels[0]]})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
