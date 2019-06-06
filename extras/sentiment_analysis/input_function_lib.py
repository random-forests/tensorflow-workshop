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
"""Input functions implementations used by sentiment_analysis.py.

You'll find 2 input function implementations:

* build_input_fn: expects preprocessed numpy data as input
  (more details in the tutorial) and will be used to train and evaluate the
  model.

* build_classify_input_fn: expects a string as input and will be used
  to classify new reviews in real time.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import rnn_common


def build_input_fn(x_in, y_in, batch_size,
                   shuffle=True, epochs=1,
                   batch_by_seq_len=False,
                   max_length=250):
  """Returns an input function created from word and class index arrays.



  Args:
    x_in: A numpy array of word indexes with shape (num_examples,
      max_sequence_length). The array is padded on the right with zeros.
    y_in: A numpy array of class indexes with shape (num_examples)
    batch_size: Batch size for the input_fn to return
    shuffle: A bool, indicating whether to shuffle the data or not.
    epochs: Number of epochs for the input fun to generate.
    batch_by_seq_len: A bool to activate sequence length batching.
    max_length: Truncate sequences longer than max_length.

  Returns:
    An `input_fn`.
  """
  def _length_bin(length, max_seq_len, length_step=10):
    """Sets the sequence length bin."""
    bin_id = (length // length_step + 1) * length_step
    return tf.cast(tf.minimum(bin_id, max_seq_len), tf.int64)

  def _make_batch(key, ds):
    """Removes extra padding and batchs the bin."""
    # eliminate the extra padding
    key = tf.cast(key, tf.int32)
    ds = ds.map(lambda x, x_len, y: (x[:key], x_len, y))

    # convert the entire contents of the bin to a batch
    ds = ds.batch(batch_size)
    return ds

  def input_fn():
    """Input function used for train and eval; usually not called directly.
    """
    # calculates the length of the sequences
    # since the inputs are already padded with zeros in the end
    # the length will be the last index that is non zero + 1
    x_len = np.array(
        [np.nonzero(seq)[0][-1] + 1 for seq in x_in]).astype('int32')

    # creates the dataset from in memory data
    # x_in: sequence of indexes that map a word to an embedding
    # x_len: sequence lengths
    # y_in: 1 if positive review, 0 if negative review
    ds = tf.contrib.data.Dataset.from_tensor_slices((x_in, x_len, y_in))

    # repeats the dataset `epochs` times
    ds = ds.repeat(epochs)

    if shuffle:
      # make sure the buffer is big enough for your data
      ds = ds.shuffle(buffer_size=25000 * 2)

    if batch_by_seq_len:
      # implement a simple `Dataset` version of `bucket_by_sequence_length`
      # https://goo.gl/y67FQm
      ds = ds.group_by_window(
          key_func=lambda x, x_len, y: _length_bin(x_len, max_length),
          reduce_func=_make_batch,
          window_size=batch_size)
    else:
      ds = ds.batch(batch_size)

    # creates iterator
    x, x_len, y = ds.make_one_shot_iterator().get_next()

    # feature must be a dictionary
    dict_x = {'x': x, rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY: x_len}
    return dict_x, y

  return input_fn


def build_classify_input_fn(review, word_to_id):
  """Returns an Input function from a string review, and a word_to_id mapping.
   The input_fn only yields a single batch before throwing an end of
   sequence error.
   The input_fn does not yield labels, so it cannot be used for training or
   evaluation.

   Args:
     review(str): A string review sentence.
     word_to_id(dict): A dict mapping words to embedding indexes.
  """
  def _word_to_index(sequence):
    """Convert a sequence of words into a sequence of indexes that map each
       word to a row in the embedding.
    """
    id_sequence = []
    UNK = 399999  # index for unknown words
    for word in sequence:
      try:
        id_sequence.append(word_to_id[word])
      except KeyError:
        id_sequence.append(UNK)  # if not in the word_to_id list set to UNK
    return np.array(id_sequence)

  def input_fn():
    """Input function used to classify new reviews manually inserted."""
    # make review a sequence of words
    review_split = review.split(' ')
    # converting words to indexes
    review_id = _word_to_index(review_split)
    # calculates the length of the sequence
    x_len = len(review_split)
    # creates the dataset from in memory data
    ds = tf.contrib.data.Dataset.from_tensors(review_id)
    # the model expects a batch
    ds = ds.batch(1)

    # creates iterator
    x = ds.make_one_shot_iterator().get_next()

    dict_x = {'x': x, rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY: [x_len]}
    # no label needed since we're only using this input function for prediction
    # if training make sure to return a label
    return dict_x, None

  return input_fn

