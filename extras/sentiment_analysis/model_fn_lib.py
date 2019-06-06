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
"""RNN Model implementation using a model function for an estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import rnn_common


def model_fn(features, labels, mode, params):
  """Returns an EstimatorSpec.

  Args:
       features(dict): a dictionary with the following keys and values.
       {
           'x': tensor,
           rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY: list
       } Where, tensor.shape == [BATCH_SIZE, SEQUENCE_LENGTH]
       and len(list) == BATCH_SIZE.
       labels: tensor of shape [BATCH_SIZE] containing integers. The value is
       0 for a negative review or 1 if it's a positive review.
       mode(str): equals to tf.estimator.ModeKeys.TRAIN,
       tf.estimator.ModeKeys.EVAL or tf.estimator.ModeKeys.PREDICT.
       params(dict): this contains hyperparameters allowing a more flexible
       implementation.
  """

  # list containing the size of each RNN (int), they will be stacked in
  # the order specified in the list
  rnn_cell_sizes = params['rnn_cell_sizes']

  # list containg the probability of applying dropout on each RNN cell
  # each value should be a float from 0 to 1
  dropout_keep_probabilities = params['dropout_keep_probabilities']

  # list containg the size of each dense layer in the model (int), they will
  # be stacked on top of the last RNN in the order specified in the list
  dnn_layer_sizes = params['dnn_layer_sizes']

  # final label dimension, since this is a classification problem this
  # is the number of classes
  label_dimension = params['label_dimension']

  # pretrained word embedding
  pretrained_embeddings = params['word_vector']

  # string, class or optimizer instance. String should be name of optimizer,
  # like 'SGD', 'Adam', 'Adagrad', ...
  # More details at: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/optimize_loss
  optimizer = params['optimizer']

  # step size used by the optimizer
  learning_rate = params['learning_rate']

  # If true the final output from the RNNs will be the average of the hidden
  # states, otherwise the output from the last cell will be used
  average_hidden_states = (params['use_hidden_states'] == 'average')

  is_training = mode == tf.estimator.ModeKeys.TRAIN
  is_eval = mode == tf.estimator.ModeKeys.EVAL
  is_predict = mode == tf.estimator.ModeKeys.PREDICT

  review = features['x']
  sequence_length = tf.cast(
      features[rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY],
      tf.int32)

  # applying pre-trained embedding
  W = tf.constant(pretrained_embeddings, name='W')
  data = tf.nn.embedding_lookup(W, review)

  if dropout_keep_probabilities:
    # if we're not training we want to keep all RNN cells
    if is_training:
      probabilities = dropout_keep_probabilities
    else:
      probabilities = [1] * len(dropout_keep_probabilities)
       
    # creating the LSTMCells and adding dropout
    # check https://www.tensorflow.org/api_docs/python/tf/contrib/rnn for more
    rnn_layers = [
        tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(size),
                                      output_keep_prob=keep_prob,
                                      state_keep_prob=keep_prob)
        for size, keep_prob in zip(rnn_cell_sizes, probabilities)
    ]

  else:
    # if not using dropout each RNN layer will consist of a regular LSTM cell
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in rnn_cell_sizes]

  # stack the layers
  multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

  # runs the RNN dynamically
  # more about it at https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
  # the output consists of a tuple with two values:
  # outputs: a tensor with shape [BATCH_SIZE, SEQUENCE_LENGTH, STATE_SIZE]
  # final state: tuple where the for each RNN layer (cell) there's a
  # tf.contrib.rnn.LSTMStateTuple where:
  # c is the hidden state and h is the output of a given cell
  # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple
  outputs, final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                           inputs=data,
                                           sequence_length=sequence_length,
                                           dtype=tf.float32)

  if average_hidden_states:
    dense_layer_input = tf.div(
        tf.reduce_sum(outputs, axis=1),
        tf.cast(sequence_length[:, tf.newaxis], dtype=tf.float32))
  else:
    # slice to keep only the last cell of the RNN
    # each value at final state is a LSTMStateTuple
    dense_layer_input = final_state[-1].h

  # construct dense layers using tf.layers
  for units in dnn_layer_sizes:
    dense_layer_input = tf.layers.dense(
        dense_layer_input, units, activation=tf.nn.relu)

  # final dense layer for prediction
  predictions = tf.layers.dense(dense_layer_input, label_dimension)
  predictions_softmax = tf.nn.softmax(predictions)

  # define model operations
  loss = None
  train_op = None
  eval_op = None

  if not is_predict:
    loss = tf.losses.sparse_softmax_cross_entropy(labels, predictions)

  if is_eval:
    eval_op = {
        'accuracy': tf.metrics.accuracy(
            tf.argmax(input=predictions_softmax, axis=1),
            labels)
    }

  if is_training:
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer=optimizer,
        learning_rate=learning_rate)

  return tf.estimator.EstimatorSpec(mode,
                                    predictions=predictions_softmax,
                                    loss=loss,
                                    train_op=train_op,
                                    eval_metric_ops=eval_op)
