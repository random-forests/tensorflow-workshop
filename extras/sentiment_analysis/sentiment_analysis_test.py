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
"""Basic test for the Sentiment Analysis tutorial related files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imdb import IMDB
from input_function_lib import build_input_fn
from model_fn_lib import model_fn
import numpy as np
import sentiment_analysis
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import rnn_common


class BaseTest(tf.test.TestCase):

  def input_fn(self):
    """Provides valid random features and labels."""
    # sequences of indexes (considering that all the sequences have length=250
    # [BATCH_SIZE, SEQUENCE_LENGTH]
    features = {
        'x': tf.random_uniform([32, 250], 0, 400000, dtype=tf.int32),
        rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY: [250] * 32
    }
    # 0: negative review, 1: positive review
    labels = tf.random_uniform([32], maxval=2, dtype=tf.int32)
    return features, labels

  def get_word_vector(self):
    """Provides a random word embedding."""
    return np.array(np.random.uniform(size=[40000, 50]), dtype=np.float32)

  def get_default_model_params(self):
    """Returns model params for default model."""
    params = {
        'rnn_cell_sizes': FLAGS.rnn_cell_sizes,
        'label_dimension': FLAGS.num_classes,
        'word_vector': self.get_word_vector(),
        'dnn_layer_sizes': FLAGS.dnn_layer_sizes,
        'optimizer': FLAGS.optimizer,
        'learning_rate': FLAGS.learning_rate,
        'dropout_keep_probabilities': FLAGS.dropout_keep_probabilities,
        'use_hidden_states': FLAGS.use_hidden_states
    }
    return params

  def model_fn_helper(self, mode, params):
    """Basic test for model_function."""

    features, labels = self.input_fn()

    spec = model_fn(features, labels, mode, params)

    predictions = spec.predictions
    self.assertAllEqual(predictions.shape[1], 2)
    self.assertEqual(predictions.dtype, tf.float32)

    if mode != tf.estimator.ModeKeys.PREDICT:
      loss = spec.loss
      self.assertAllEqual(loss.shape, ())
      self.assertEqual(loss.dtype, tf.float32)

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = spec.train_op
      self.assertAllEqual(train_op.shape, ())
      self.assertEqual(train_op.dtype, tf.float32)

    if mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = spec.eval_metric_ops
      self.assertAllEqual(eval_metric_ops['accuracy'][0].shape, ())
      self.assertAllEqual(eval_metric_ops['accuracy'][1].shape, ())
      self.assertEqual(eval_metric_ops['accuracy'][0].dtype, tf.float32)
      self.assertEqual(eval_metric_ops['accuracy'][1].dtype, tf.float32)

  def test_model_fn_train_mode(self):
    """Basic test for train mode."""
    params = self.get_default_model_params()
    self.model_fn_helper(tf.estimator.ModeKeys.TRAIN, params)

  def test_model_fn_eval_mode(self):
    """Basic test for eval mode."""
    params = self.get_default_model_params()
    self.model_fn_helper(tf.estimator.ModeKeys.EVAL, params)

  def test_model_fn_predict_mode(self):
    """Basic test for predict mode."""
    params = self.get_default_model_params()
    self.model_fn_helper(tf.estimator.ModeKeys.PREDICT, params)

  def test_input_fn(self):
    """Basic test for input function."""
    imdb = IMDB('data')
    x_train, y_train, _, _ = imdb.get_data()
    input_fn = build_input_fn(x_train, y_train, 32, epochs=1)

    features, labels = input_fn()
    # shape
    self.assertAllEqual(features['x'][0].shape, (250,))
    self.assertAllEqual(
        features[rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY][0].shape,
        ()
    )
    # type
    self.assertAllEqual(features['x'][0].dtype, tf.int32)
    self.assertAllEqual(
        features[rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY][0].dtype,
        tf.int32
    )
    self.assertAllEqual(labels.dtype, tf.int64)


if __name__ == '__main__':
  FLAGS = sentiment_analysis.parser.parse_args()
  tf.logging.set_verbosity(tf.logging.ERROR)  # enable TensorFlow logs
  tf.test.main()
