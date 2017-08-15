# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# ==============================================================================
"""A helper class for fetching and importing the IMDB dataset.

This helper will download the data available at
https://github.com/adeshpande3/LSTM-Sentiment-Analysis that is a preprocessed
version of the Large Movie Review Dataset available at
http://ai.stanford.edu/~amaas/data/sentiment/. Here you'll also
find functions to access this data once it's available.
"""

import os
import tarfile

import numpy as np
from six.moves import urllib


class IMDB(object):
  """A helper class for fetching and importing the IMDB dataset.

  The three `get` methods each import an component of data
  from the downloaded files.
  """

  def __init__(self, data_path, percentage_train=0.9):
    """Create an IMDB data loader.
    Args:
      data_path: Where to store the downloaded files.
      percentage_train: The fraction of the dataset set to use for training.
    """
    # path where the data will be stored
    self.data_path = data_path
    # postive reviews will have label 1, and negative reviews label 0
    self._POS = 1
    self._NEG = 0
    # path to where data is hosted
    self._DATA_URL = 'https://github.com/adeshpande3/LSTM-Sentiment-Analysis/blob/master/training_data.tar.gz?raw=true'
    # perecentage of data used for training
    self._PERCENTAGE_TRAIN = percentage_train
    # if data is not in data_path download it from _DATA_URL
    self._maybe_download()

  def _get_word_list(self):
    """Returns list with words available in the word embedding."""
    return list(np.load(os.path.join(self.data_path, 'wordsList.npy')))

  def get_word_to_index(self):
    """Returns dict mapping a word to an index in the word embedding."""
    word_list = self._get_word_list()
    word_dict = {word_list[i]: i for i in range(len(word_list))}
    return word_dict

  def get_index_to_word(self):
    """Returns dict mapping an index to a word in the word embedding."""
    word_list = self._get_word_list()
    word_dict = {i: word_list[i] for i in range(len(word_list))}
    return word_dict

  def get_word_vector(self):
    """Returns the pretrained word embedding."""
    return np.load(os.path.join(self.data_path, 'wordVectors.npy'))

  def get_data(self):
    """Returns the preprocessed IMDB dataset for training and evaluation.

    The data contain 25000 reviews where the first half is positive and the
    second half is negative. This function by default will return 90% of the
    data as training data and 10% as evaluation data.
    """

    data = np.load(os.path.join(self.data_path, 'idsMatrix.npy'))
    # the first half of the data length are positive reviews
    # the other half are negative reviews
    data_len = data.shape[0]
    label = np.array(
        [self._POS if i < data_len/2 else self._NEG for i in range(data_len)]
    )

    # shuffle the data
    p = np.random.permutation(data_len)
    shuffled_data = data[p]
    shuffled_label = label[p]

    # separate training and evaluation
    train_limit = int(data_len * self._PERCENTAGE_TRAIN)

    train_data = shuffled_data[:train_limit]
    train_label = shuffled_label[:train_limit]
    eval_data = shuffled_data[train_limit:]
    eval_label = shuffled_label[train_limit:]

    return train_data, train_label, eval_data, eval_label

  def _maybe_download(self):
    """Maybe downloads data available at https://github.com/adeshpande3/LSTM-Sentiment-Analysis."""
    try:
      self.get_word_to_index()
      self.get_word_vector()
      self.get_data()
    except IOError:
      print('Data is not available at %s, Downloading it...' % self.data_path)
      # if the data_path does not exist we'll create it
      if not os.path.exists(self.data_path):
        os.makedirs(self.data_path)

      # download data
      tar_path = os.path.join(self.data_path, 'data.tar.gz')
      urllib.request.urlretrieve(self._DATA_URL, tar_path)
      # extract data and save at self.data_path
      tar = tarfile.open(tar_path)
      tar.extractall(self.data_path)
      tar.close()

      print('Download complete!')

