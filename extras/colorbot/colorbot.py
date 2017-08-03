# coding: utf-8

# # Colorbot

# ## Dependencies
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import rnn_common
from tensorflow.python.estimator import model_fn as model_fn_lib


parser = argparse.ArgumentParser(prog='Play with Colorbot!')

# Arguments
parser.add_argument('--model_dir', type=str, default='pretrained',
                    help='Local path to the folder where the colorbot'
                         'model is or should be saved if running on training'
                         ' mode.')

parser.add_argument('--mode', type=str, default="classify",
                    help='Local path to the folder where the colorbot'
                         'model is.')

# Data files
TRAIN_INPUT = 'data/train.csv'
TEST_INPUT = 'data/test.csv'
MY_TEST_INPUT = 'data/mytest.csv'

# Parameters for training
BATCH_SIZE = 64

# Parameters for data processing
VOCAB_SIZE = 256
CHARACTERS = [chr(i) for i in range(VOCAB_SIZE)]
SEQUENCE_LENGTH_KEY = 'sequence_length'
COLOR_NAME_KEY = 'color_name'


# ## Helper functions
# Returns the column values from a CSV file as a list
def _get_csv_column(csv_file, column_name):
  with open(csv_file, 'r') as f:
    df = pd.read_csv(f)
    return df[column_name].tolist()


# Plot a color image
def _plot_rgb(rgb, color_name):
  data = [[rgb]]
  plt.imshow(data, interpolation='nearest')
  plt.title(color_name)
  plt.show()


# Creating a input function for training and evaluation
def get_input_fn(csv_file, batch_size, num_epochs=1, shuffle=True):
  def _parse(line):
    # each line: name, red, green, blue
    # split line
    items = tf.string_split([line], ',').values

    # get color (r, g, b)
    color = tf.string_to_number(items[1:], out_type=tf.float32) / 255.0

    # split color_name into a sequence of characters
    color_name = tf.string_split([items[0]], '')
    # length = index of last char + 1
    length = color_name.indices[-1, 1] + 1
    color_name = color_name.values

    return color, color_name, length

  def _length_bin(length, cast_value=5, max_bin_id=10):
    # Chooses a bin for a word given it's length.
    # The goal is to use group_by_window to group words
    # with the ~ same ~ length in the same bin.
    # Each bin will have the size of a batch, so it can train faster.
    bin_id = tf.cast(length / cast_value, dtype=tf.int64)
    return tf.minimum(bin_id, max_bin_id)

  def _pad_batch(ds, batch_size):
    return ds.padded_batch(batch_size,
                           padded_shapes=([None], [None], []),
                           padding_values=(0.0, chr(0), tf.cast(0, tf.int64)))

  def input_fn():
    # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data
    dataset = (
        tf.contrib.data.TextLineDataset(csv_file)  # reading from the HD
        .skip(1)  # skip header
        .repeat(num_epochs)  # repeat dataset the number of epochs
        .map(_parse)  # parse text to variables
        # batch by sorted length
        .group_by_window(
            key_func=lambda color, color_name, length: _length_bin(length),
            reduce_func=lambda key, ds: _pad_batch(ds, batch_size),
            window_size=batch_size)
    )

    # for our "manual" test we don't want to shuffle the data
    if shuffle:
      dataset = dataset.shuffle(buffer_size=100000)

    # create iterator
    color, color_name, length = dataset.make_one_shot_iterator().get_next()

    features = {
        COLOR_NAME_KEY: color_name,
        SEQUENCE_LENGTH_KEY: length,
    }

    return features, color
  return input_fn

# Creating an input function for prediction
def get_pred_input_fn(color):
  def input_fn():
    seq_len = len(color)
    # color is now a sequence of chars
    color_split = tf.string_split([color], '').values

    # creating dataset
    dataset = tf.contrib.data.Dataset.from_tensors((color_split))
    # generating a batch, so it has the right rank
    dataset = dataset.batch(1)

    # creating iterator
    color_name = dataset.make_one_shot_iterator().get_next()

    features = {
        COLOR_NAME_KEY: color_name,
        SEQUENCE_LENGTH_KEY: [seq_len]
    }

    # we're just predicting, so the label can be None
    # if you're training make sure to return a label
    return features, None
  return input_fn


train_input_fn = get_input_fn(TRAIN_INPUT, BATCH_SIZE)
test_input_fn = get_input_fn(TEST_INPUT, BATCH_SIZE)


# ## Creating the Estimator model
def get_model_fn(rnn_cell_sizes,
                 label_dimension,
                 dnn_layer_sizes=[],
                 optimizer='SGD',
                 learning_rate=0.01):

  def _model_fn(features, labels, mode):

    color_name = features[COLOR_NAME_KEY]
    # int64 -> int32
    sequence_length = tf.cast(features[SEQUENCE_LENGTH_KEY], dtype=tf.int32)

    # ----------- Preparing input --------------------
    # Creating a tf constant to hold the map char -> index
    mapping = tf.constant(CHARACTERS, name='mapping')
    table = tf.contrib.lookup.index_table_from_tensor(mapping, dtype=tf.string)
    int_color_name = table.lookup(color_name)

    # representing colornames with one hot representation
    color_name_onehot = tf.one_hot(int_color_name, depth=len(CHARACTERS) + 1)

    # ---------- RNN -------------------
    # Each RNN layer will consist of a LSTM cell
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in rnn_cell_sizes]

    # Construct the layers
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # Runs the RNN model dynamically
    # more about it at:
    # https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
    outputs, final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                             inputs=color_name_onehot,
                                             sequence_length=sequence_length,
                                             dtype=tf.float32)

    # Slice to keep only the last cell of the RNN
    last_activations = rnn_common.select_last_activations(outputs,
                                                          sequence_length)

    # ------------ Dense layers -------------------
    # Construct dense layers on top of the last cell of the RNN
    for units in dnn_layer_sizes:
      last_activations = tf.layers.dense(
          last_activations, units, activation=tf.nn.relu)

    # Final dense layer for prediction
    predictions = tf.layers.dense(last_activations, label_dimension)

    # ----------- Loss and Optimizer ----------------
    loss = None
    train_op = None

    if mode != tf.estimator.ModeKeys.PREDICT:
      loss = tf.losses.mean_squared_error(labels, predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = tf.contrib.layers.optimize_loss(
          loss,
          tf.contrib.framework.get_global_step(),
          optimizer=optimizer,
          learning_rate=learning_rate)

    return model_fn_lib.EstimatorSpec(mode, predictions=predictions,
                                      loss=loss, train_op=train_op)
  return _model_fn

# Creating model_fn
model_fn = get_model_fn(rnn_cell_sizes=[256, 128],  # size of the hidden layers
                        label_dimension=3,  # since is RGB
                        # size of units in the dense layers on top of the RNN
                        dnn_layer_sizes=[128],
                        optimizer='Adam',  # changing optimizer to Adam
                        learning_rate=0.01)

# Getting arguments
FLAGS = parser.parse_args()

# Creates estimator on model_dir
estimator = tf.estimator.Estimator(model_fn=model_fn,
                                            model_dir=FLAGS.model_dir)

if FLAGS.mode == "train":
  NUM_EPOCHS = 40
  for i in range(NUM_EPOCHS):
    print('Training epoch %d' % i)
    print('-' * 20)
    estimator.train(input_fn=train_input_fn)
    print('Evaluating epoch %d' % i)
    print('-' * 20)
    evaluation = estimator.evaluate(input_fn=test_input_fn)
    print(evaluation)

# Making Predictions
print('Colorbot is ready to generate colors!')

EXIT_COMMAND = '<exit>'
while True:
  color_name = raw_input('give me a color name (or %s): ' % (EXIT_COMMAND))
  if color_name == EXIT_COMMAND:
    break

  print('Generating color...')
  preds = estimator.predict(input_fn=get_pred_input_fn(color_name))
  for p, name in zip(preds, [color_name]):
    color = tuple(map(int, p * 255))
    _plot_rgb(p, name)
