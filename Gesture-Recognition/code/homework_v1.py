from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import cv2
import time

# Numpy
import numpy as np

# Tensorflow
import tensorflow as tf

# Sklearn
from sklearn.model_selection import train_test_split

# Datetime
from datetime import datetime


# -----------------------------------------------------------------------------
# Two Layer Fc
# -----------------------------------------------------------------------------

def inference(images, image_pixels, hidden_units, classes, reg_constant=0):
    '''Build the model up to where it may be used for inference.

    Args:
        images: Images placeholder (input data).
        image_pixels: Number of pixels per image.
        hidden_units: Size of the first (hidden) layer.
        classes: Number of possible image classes/labels.
        reg_constant: Regularization constant (default 0).

    Returns:
        logits: Output tensor containing the computed logits.
    '''

    # Layer 1
    with tf.variable_scope('Layer1'):
        # Define the variables
        weights = tf.get_variable(
            name='weights',
            shape=[image_pixels, hidden_units],
            initializer=tf.truncated_normal_initializer(
                stddev=1.0 / np.sqrt(float(image_pixels))),
            regularizer=tf.contrib.layers.l2_regularizer(reg_constant)
        )

        biases = tf.Variable(tf.zeros([hidden_units]), name='biases')

        # Define the layer's output
        hidden = tf.nn.relu(tf.matmul(images, weights) + biases)

    # Layer 2
    with tf.variable_scope('Layer2'):
        # Define variables
        weights = tf.get_variable('weights', [hidden_units, classes],
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=1.0 / np.sqrt(float(hidden_units))),
                                  regularizer=tf.contrib.layers.l2_regularizer(reg_constant))

        biases = tf.Variable(tf.zeros([classes]), name='biases')

        # Define the layer's output
        logits = tf.matmul(hidden, weights) + biases

        # Define summery-operation for 'logits'-variable
        tf.summary.histogram('logits', logits)

    return logits


def loss(logits, labels):
    '''Calculates the loss from logits and labels.

    Args:
      logits: Logits tensor, float - [batch size, number of classes].
      labels: Labels tensor, int64 - [batch size].

    Returns:
      loss: Loss tensor of type float.
    '''

    with tf.name_scope('Loss'):
        # Operation to determine the cross entropy between logits and labels
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross_entropy'))

        # Operation for the loss function
        loss = cross_entropy + tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES))

        # Add a scalar summary for the loss
        tf.summary.scalar('loss', loss)

    return loss


def training(loss, learning_rate):
    '''Sets up the training operation.

    Creates an optimizer and applies the gradients to all trainable variables.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_step: The op for training.
    '''

    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create a gradient descent optimizer
    # (which also increments the global step counter)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)

    return train_step


def evaluation(logits, labels):
    '''Evaluates the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch size, number of classes].
      labels: Labels tensor, int64 - [batch size].

    Returns:
      accuracy: the percentage of images where the class was correctly predicted.
    '''

    with tf.name_scope('Accuracy'):
        # Operation comparing prediction with true label
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)

        # Operation calculating the accuracy of the predictions
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Summary operation for the accuracy
        tf.summary.scalar('train_accuracy', accuracy)

    return accuracy


# -----------------------------------------------------------------------------
# Dataset Loader & Classifier & Prediction
# -----------------------------------------------------------------------------

FILE_PATH = '../original_images_v1'
CLASSES = ['1', '2', '3', '4', '5', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S',
           'T', 'U', 'W', 'Y', 'Z']


# List images of class in file path
def list_folder_contents(filepath, class_name):
    result = []
    files = os.listdir(filepath)

    for file in files:
        if re.search(r'^' + class_name, file):
            file_name = os.path.join('%s/%s' % (filepath, file))
            result.append(file_name)

    return result


# Init dataset
# Item in dataset is dict: {image, label}
def load_image():
    result = []

    for index in range(len(CLASSES)):
        class_name = CLASSES[index]
        file_names = list_folder_contents(FILE_PATH, class_name)
        for file_name in file_names:
            # Read image
            image = cv2.imread(file_name, cv2.IMREAD_COLOR)

            # Filter skin
            image_skin = filter_skin(resize_image(image))

            # Get image outline
            image_outline = get_image_outline(image_skin)

            # cv2.imshow('image', image_outline)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Save image in folder
            save_image(file_name.replace('../original_images_v1/', ''), image_outline)

            vector = image_outline.flatten()
            result.append({
                'image': vector,
                'label': index
            })

            print('Load image ' + file_name + ' done')

    result = format_dataset(result)

    print(result['images_train'].shape)
    print(result['labels_train'].shape)
    print(result['images_test'].shape)
    print(result['labels_test'].shape)

    return result


# Save processed image
def save_image(file_name, image):
    path = '../images_v1'
    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        os.mkdir(path)

    cv2.imencode('.jpg', image)[1].tofile(abs_path + '\\' + file_name)
    print('Save image ' + file_name + ' done')


# Get image outline
def get_image_outline(image):
    # Build kernel 3*3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Dilate process
    dilate = cv2.dilate(image, kernel)
    # Erode process
    erode = cv2.erode(image, kernel)
    # Absdiff process
    absdiff = cv2.absdiff(dilate, erode)
    # threshold process
    retval, image_threshold = cv2.threshold(absdiff, 127, 255, cv2.THRESH_BINARY)
    # Bitwise_not process
    image_bitwise_not = cv2.bitwise_not(image_threshold)
    # MedianBlur process
    image_median_blur = cv2.medianBlur(image_bitwise_not, 3)

    return image_median_blur


# Filter skin
def filter_skin(image):
    image_skin = image.copy()
    m, n, k = image_skin.shape

    for i in range(m):
        for j in range(n):
            b = image_skin.item(i, j, 0)
            g = image_skin.item(i, j, 1)
            r = image_skin.item(i, j, 2)
            if not is_skin(r, g, b):
                image_skin.itemset((i, j, 0), 255)
                image_skin.itemset((i, j, 1), 255)
                image_skin.itemset((i, j, 2), 255)

    return image_skin


# Determine whether the skin
def is_skin(r, g, b):
    result = False

    if (abs(r - g) > 15) and (r > g) and (r > b):
        if (r > 95) and (g > 40) and (b > 20) and (max(r, g, b) - min(r, g, b) > 15):
            result = True
        elif (r > 220) and (g > 210) and (b > 170):
            result = True

    return result


# Resize image
def resize_image(image):
    return cv2.resize(image, (200, 200))


# Format dataset
def format_dataset(ds):
    sample = []
    target = []

    for item in ds:
        sample.append(item['image'])
        target.append(item['label'])

    images_train, images_test, labels_train, labels_test = train_test_split(sample,
                                                                            target,
                                                                            test_size=0.33,
                                                                            random_state=42)

    return {
        'images_train': np.array(images_train),
        'labels_train': np.array(labels_train),
        'images_test': np.array(images_test),
        'labels_test': np.array(labels_test),
        'classes': CLASSES,
        'sample': sample,
        'target': target
    }


def gen_batch(data, batch_size, num_iter):
    data = np.array(data)
    index = len(data)
    for i in range(num_iter):
        index += batch_size
        if (index + batch_size) > len(data):
            index = 0
            shuffled_indices = np.random.permutation(np.arange(len(data)))
            data = data[shuffled_indices]
        yield data[index:index + batch_size]


dataset = load_image()

# -----------------------------------------------------------------------------
# Run Fc Model
# -----------------------------------------------------------------------------

# Model parameters as external flags
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the training.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 1500, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide dataset sizes without remainder.')
flags.DEFINE_string('train_dir', '../log', 'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')
FLAGS._parse_flags()

print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{} = {}'.format(attr, value))
print()

IMAGE_PIXELS = 200 * 200 * 3

beginTime = time.time()

# Put logs for each run in separate directory
logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

# Uncommenting these lines removes randomness
# You'll get exactly the same result on each run
# np.random.seed(1)
# tf.set_random_seed(1)

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')

# Operation for the classifier's result
logits = inference(images_placeholder, IMAGE_PIXELS, FLAGS.hidden1, len(CLASSES), reg_constant=FLAGS.reg_constant)

# Operation for the loss function
loss = loss(logits, labels_placeholder)

# Operation for the training step
train_step = training(loss, FLAGS.learning_rate)

# Operation calculating the accuracy of our predictions
accuracy = evaluation(logits, labels_placeholder)

# Operation merging summary data for TensorBoard
summary = tf.summary.merge_all()

# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------

with tf.Session() as sess:
    # Initialize variables and create summary-writer
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)

    # Generate input data batches
    zipped_data = zip(dataset['images_train'], dataset['labels_train'])
    batches = gen_batch(list(zipped_data), FLAGS.batch_size, FLAGS.max_steps)

    for i in range(FLAGS.max_steps):

        # Get next input data batch
        batch = next(batches)
        images_batch, labels_batch = zip(*batch)
        feed_dict = {
            images_placeholder: images_batch,
            labels_placeholder: labels_batch
        }

        # Periodically print out the model's current accuracy
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)

        # Perform a single training step
        sess.run([train_step, loss], feed_dict=feed_dict)

    # After finishing the training, evaluate on the test set
    test_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: dataset['images_test'],
        labels_placeholder: dataset['labels_test']})
    print('Test accuracy {:g}'.format(test_accuracy))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))
