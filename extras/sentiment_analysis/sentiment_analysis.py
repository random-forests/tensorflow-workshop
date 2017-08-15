# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train and evaluate a RNN Model used for Sentiment Analysis."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import re

from imdb import IMDB
from input_function_lib import build_classify_input_fn
from input_function_lib import build_input_fn
from model_fn_lib import model_fn
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner  # run an experiment
print('TensorFlow version', tf.__version__)


parser = argparse.ArgumentParser()


# script related args
parser.add_argument(
    '--mode', type=str, choices=['train', 'classify'], default='train',
    help='This defines how you want to execute this script.\n'
         'train: train and eval a new model and save it on model_dir\n'
         'classify: predict new reviews with a pretrained model using'
         ' the standard IO')

parser.add_argument(
    '--data_path', type=str, default='data',
    help='Path to where the data should be downloaded.')

# tensorflow related args
parser.add_argument(
    '--model_dir', type=str, default='sentiment_analysis_output',
    help='The directory where the model outputs should be stored.')

parser.add_argument('--run_experiment', dest='run_experiment',
                    action='store_true')
parser.add_argument('--dont_run_experiment', dest='run_experiment',
                    action='store_false')
parser.set_defaults(run_experiment=True)

# IO related args
parser.add_argument('--batch_by_seq_len', dest='batch_by_seq_len',
                    action='store_true')
parser.add_argument('--dont_batch_by_seq_len', dest='batch_by_seq_len',
                    action='store_false')
parser.set_defaults(batch_by_seq_len=True)

parser.add_argument(
    '--train_batch_size', type=int, default=32,
    help='Batch size used for training.')

parser.add_argument(
    '--eval_batch_size', type=int, default=32,
    help='Batch size used for evaluation.')

parser.add_argument(
    '--sample_input_size', type=int, default=5,
    help='Number of examples to be used for prediction.'
         'Those will be randomly chosen from the evaluation dataset.')

# training related args
parser.add_argument(
    '--num_epochs', type=int, default=8,
    help='Num epochs used for training (for evaluation is always 1).')

# tunning model
parser.add_argument(
    '--optimizer', type=str, default='Adam',
    help='Optimizer used for training.')

parser.add_argument(
    '--learning_rate', type=int, default=0.001,
    help='Learning rate.')

parser.add_argument('--use_hidden_states', type=str,
                    choices=['last', 'average'], default='last',
                    help='By default it will average the hidden states'
                         ' as describe in the paper linked in the tutorial.'
                         ' Otherwise will just consider the last output.')

parser.add_argument(
    '--rnn_cell_sizes', nargs='+', type=int, default=[128],
    help='Size of the hidden state for each RNN cell.')

parser.add_argument(
    '--dnn_layer_sizes', nargs='+', type=int, default=[],
    help='Size of the hidden state for each RNN cell.')

parser.add_argument(
    '--dropout_keep_probabilities', nargs='+', type=float,
    default=[],
    help='Dropout probabilities to keep the cell. '
         'If provided should have the same length '
         'as rnn_cell_sizes.')

# model specific args
parser.add_argument(
    '--num_classes', type=int, default=2,
    help='Number of output classes. '
         'For sentiment analysis is 2 (positive and negative)')


def ids_to_sentence(sequence, id_to_word):
  """Given a sequence of numbers returns a string that represents it."""
  return ' '.join(id_to_word[index] for index in sequence)


def get_sample_data(x_dataset, y_dataset, sample_size):
  """Randomly chooses sample_size elements from x_dataset and y_dataset."""
  indexes = np.random.randint(x_dataset.shape[0],
                              size=sample_size)

  return x_dataset[indexes], y_dataset[indexes]


def build_experiment_fn(estimator, train_input, eval_input):
  """Return an Experiment function."""
  def _experiment_fn(run_config, hparams):
    """Create experiment.

    Experiments perform training on several workers in parallel. In other
    words Experiments know how to invoke train and eval in a sensible
    fashion for distributed training.

    We first prepare an estimator, and bundle it together with input functions
    for training and evaluation then collect all that in an Experiment object
    that will train and evaluate our model.
    """
    del run_config, hparams  # unused args
    return tf.contrib.learn.Experiment(
        estimator,
        train_input_fn=train_input,
        eval_input_fn=eval_input
    )
  return _experiment_fn


def validate_args(FLAGS):
  """Validate arguments."""
  len_keep_prob = len(FLAGS.dropout_keep_probabilities)
  if len_keep_prob > 0 and len_keep_prob != len(FLAGS.rnn_cell_sizes):
    raise ValueError('If using dropout dropout_keep_probabilites '
                     'must have the same length as FLAGS.rnn_cell_sizes')


def print_instructions():
  """Print input instructions for classifaction mode."""
  print('INSTRUCTIONS')
  print('=' * 40)
  print('The model expects lower case letters or numbers as input'
        ' (no special puntuaction will be removed and upper case letters will'
        '  be converted to lower case.')
  print('Review Example: this is a good movie i cant believe')
  print('=' * 40)
  print('If you see this warning message: "Input graph does not contain a'
        ' Queue Runner..." ignore it; it is not fatal.')


def format_input(review):
  """Remove non-alphanumeric chars and turns the chars to lower case."""
  pattern = re.compile(r'([^\s\w]|_)+')
  return pattern.sub('', review).lower()


def main(unused_argv):
  # validate args
  FLAGS = parser.parse_args()
  validate_args(FLAGS)
  print(FLAGS)

  # get the data from https://github.com/adeshpande3/LSTM-Sentiment-Analysis.
  print('Getting data...')

  imdb = IMDB(FLAGS.data_path)
  x_train, y_train, x_eval, y_eval = imdb.get_data()

  print('Size of the train dataset:', x_train.shape[0])
  print('Size of the eval dataset:', x_eval.shape[0])

  # creating sample dataset from the evaluation data
  # used only for visualization
  x_sample, y_sample = get_sample_data(x_eval, y_eval, FLAGS.sample_input_size)

  # creating run config
  run_config = tf.contrib.learn.RunConfig(model_dir=FLAGS.model_dir)
  # define model parameters
  model_params = {
      'rnn_cell_sizes': FLAGS.rnn_cell_sizes,
      'label_dimension': FLAGS.num_classes,
      'word_vector': imdb.get_word_vector(),
      'dnn_layer_sizes': FLAGS.dnn_layer_sizes,
      'optimizer': FLAGS.optimizer,
      'learning_rate': FLAGS.learning_rate,
      'dropout_keep_probabilities': FLAGS.dropout_keep_probabilities,
      'use_hidden_states': FLAGS.use_hidden_states
  }

  # creating estimator
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     config=run_config,
                                     params=model_params)

  if FLAGS.mode == 'train':
    # defining input functions
    # train input function
    train_input_fn = build_input_fn(x_train, y_train, FLAGS.train_batch_size,
                                    epochs=FLAGS.num_epochs,
                                    batch_by_seq_len=FLAGS.batch_by_seq_len)

    # eval input function
    eval_input_fn = build_input_fn(x_eval, y_eval, FLAGS.eval_batch_size,
                                   epochs=1)

    # input function used to classify samples
    sample_input_fn = build_input_fn(x_sample, y_sample, 1, epochs=1,
                                     shuffle=False)

    if FLAGS.run_experiment:
      # run training and evaluation
      learn_runner.run(
          build_experiment_fn(estimator, train_input_fn, eval_input_fn),
          run_config=run_config
      )
      pass
    else:
      # training
      estimator.train(input_fn=train_input_fn)

      # evalutaion
      estimator.evaluate(input_fn=eval_input_fn)

    # since we have a small number of predictions we're converting it to a
    # list, when running `predict` on a big dataset is better to iterate in
    # the predictions instead
    predictions = list(estimator.predict(input_fn=sample_input_fn))

    # loading map from index to word
    index_to_word = imdb.get_index_to_word()

    # printing movie review, prediction and label
    # for visualization
    for i in range(FLAGS.sample_input_size):
      print(ids_to_sentence(x_sample[i], index_to_word))
      print('Prediction:', predictions[i])
      print('Label:', y_sample[i])

  elif FLAGS.mode == 'classify':
    # loading map from word to index
    word_to_index = imdb.get_word_to_index()

    print_instructions()
    while True:
      try:
        review = format_input(raw_input('Write your review: '))
      except EOFError:
        break

      print('Your review:', review)
      print('Generating prediction...')
      preds = estimator.predict(input_fn=build_classify_input_fn(review,
                                                                 word_to_index))
      for p in preds:
        print('Negative:', p[0])
        print('Positive:', p[1])


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)  # enable TensorFlow logs
  tf.app.run()
