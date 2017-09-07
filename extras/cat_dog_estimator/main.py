"""
A very simplified introduction to TensorFlow using Estimator API for training a
cat vs. dog classifier from the CIFAR-10 dataset. This version is intentionally
simplified and has a lot of room for improvment, in speed and accuracy.
"""

import sys

import numpy as np
from PIL import Image
import tensorflow as tf

DATA_FILE = 'catdog_data.npy'
NUM_IMAGES = 10000
MODEL_DIR = 'models'

# Some of the tunable hyperparameters are set here
LEARNING_RATE = 0.01
MOMENTUM = 0.9
TRAIN_EPOCHS = 10
BATCH_SIZE = 32

def model_fn(features, labels, mode):
  """Defines the CNN model that runs on the data.

    The model we run is 3 convolutional layers followed by 1 fully connected
    layer before the output. This is much simpler than most CNN models and is
    designed to run decently on CPU. With a GPU, it is possible to scale to
    more layers and more filters per layer.

    Args:
        features: batch_size x 32 x 32 x 3 uint8 images
        labels: batch_size x 1 uint8 labels (0 or 1)
        mode: TRAIN, EVAL, or PREDICT

    Returns:
        EstimatorSpec which defines the model to run
  """

  # Preprocess the features by converting to floats in [-0.5, 0.5]
  features = tf.cast(features, tf.float32)
  features = (features / 255.0) - 1.0

  # Define the CNN network
  # conv1: 32 x 32 x 3 -> 32 x 32 x 16
  net = tf.layers.conv2d(
      inputs=features,
      filters=16,                 # 16 channels after conv
      kernel_size=3,              # 3x3 conv kernel
      padding='same',             # Output tensor is same shape
      activation=tf.nn.relu)      # ReLU activation

  # pool1: 32 x 32 x 16 -> 16 x 16 x 16
  net = tf.layers.max_pooling2d(
      inputs=net,
      pool_size=2,
      strides=2)                  # Downsample 2x

  # conv2: 16 x 16 x 16 -> 16 x 16 x 32
  net = tf.layers.conv2d(
      inputs=net,
      filters=32,
      kernel_size=3,
      padding='same',
      activation=tf.nn.relu)

  # pool2: 16 x 16 x 32 -> 8 x 8 x 32
  net = tf.layers.max_pooling2d(
      inputs=net,
      pool_size=2,
      strides=2)

  # conv3: 8 x 8 x 32 -> 8 x 8 x 64
  net = tf.layers.conv2d(
      inputs=net,
      filters=64,
      kernel_size=3,
      padding='same',
      activation=tf.nn.relu)

  # flat: 8 x 8 x 64 -> 4096
  net = tf.contrib.layers.flatten(net)

  # fc4: 4096 -> 1000
  net = tf.layers.dense(
      inputs=net,
      units=1000,
      activation=tf.nn.relu)

  # output: 1000 -> 2
  logits = tf.layers.dense(
      inputs=net,
      units=2)

  # Softmax for probabilities
  probabilities = tf.nn.softmax(logits)

  predictions = tf.argmax(
      input=logits,
      axis=1,
      output_type=tf.int32)

  # Return maximum prediction if we're running PREDICT
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'prediction': predictions,
            'probability': probabilities})

  # Loss function and optimizer for training
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=tf.one_hot(labels, depth=2),
      logits=logits)

  train_op = tf.train.MomentumOptimizer(
      LEARNING_RATE, MOMENTUM).minimize(
          loss=loss,
          global_step=tf.train.get_global_step())

  # Accuracy for evaluation
  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels,
          predictions=predictions)}

  # EVAL uses loss and eval_metric_ops, TRAIN uses loss and train_op
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)

def input_fn_wrapper(is_training):
  """Input function wrapper for training and eval.

  A wrapper funcution is used because we want to have slightly different
  behavior for the dataset during training (shuffle and loop data) and
  evaluation (don't shuffle and run exactly once).

  Args:
      is_training: bool for if the model is training

  Returns:
      function with signature () -> features, labels
      where features and labels are the same shapes expected by model_fn
  """
  def input_fn():
    data = np.load(DATA_FILE).item()
    np_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': data['images']},
        y=data['labels'],
        batch_size=BATCH_SIZE,
        shuffle=is_training,
        num_epochs=None if is_training else 1)

    features_dict, labels = np_input_fn()
    # Since the only feature is the image itself, return the image directly
    # instead of the features dict
    return features_dict['x'], labels

  return input_fn

def process_image(image_file):
  """Convert PIL Image to a format that the network can accept.

  Operations performed:
      - Load image file
      - Central crop square
      - Resize to 32 x 32
      - Convert to numpy array

  Args:
      image_file: str file name of image

  Returns:
      numpy.array image which shape [1, 32, 32, 3]

  Assumes that image is RGB and at least 32 x 32.
  """
  image = Image.open(image_file)
  width, height = image.size

  min_dim = min(width, height)
  left = (width - min_dim) / 2
  top = (height - min_dim) / 2
  right = (width + min_dim) / 2
  bottom = (height + min_dim) / 2

  image = image.crop((left, top, right, bottom))
  image = image.resize((32, 32), resample=Image.BILINEAR)
  image = np.asarray(image, dtype=np.uint8)
  image = np.reshape(image, [1, 32, 32, 3])

  return image

def main():
  if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'predict']:
    print 'Usage: python main.py [train|predict] [predict file]'
    sys.exit()

  tf.logging.set_verbosity(tf.logging.INFO)
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=tf.estimator.RunConfig().replace(
          model_dir=MODEL_DIR))
  if sys.argv[1] == 'train':
    steps_per_epoch = NUM_IMAGES / BATCH_SIZE       # Rounds down
    for epoch in xrange(TRAIN_EPOCHS):
      estimator.train(
          input_fn=input_fn_wrapper(True),
          steps=steps_per_epoch)
      # Evaluating on the same dataset as training for simplicity, normally
      # this is a very bad idea since you are not testing how well your
      # model generalizes to unseen data.
      estimator.evaluate(input_fn=input_fn_wrapper(False))
  else:   # sys.argv[1] == 'predict'
    if len(sys.argv) < 3:
      print 'Usage: python main.py predict [predict file]'
      sys.exit()
    image = process_image(sys.argv[2])

    # Define a new input function for prediction which output a single image
    def predict_input_fn():
      np_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={'x': image},
          num_epochs=1,
          shuffle=False)
      features_dict = np_input_fn()
      return features_dict['x']

    pred_dict = estimator.predict(
        input_fn=predict_input_fn).next()

    print 'Probability of cat: %.5f\tProbability of dog: %.5f' % (
        pred_dict['probability'][1], pred_dict['probability'][0])
    print 'Prediction %s' % ('CAT' if pred_dict['prediction'] == 1 else 'DOG')

if __name__ == '__main__':
  main()

