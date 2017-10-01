"""One-time script for extracting all the cat and dog images from CIFAR-10."""

import cPickle
import numpy as np
from PIL import Image

TRAIN_FILES = ['cifar-10-batches-py/data_batch_%d' % i for i in range(1,6)]
TEST_FILE = 'test_batch'

CAT_INPUT_LABEL = 3
DOG_INPUT_LABEL = 5

CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = cPickle.load(fo)
    return dict

data = []

# Count number of cats/dogs
num_cats = 0
num_dogs = 0

for data_file in TRAIN_FILES:
  d = unpickle(data_file)
  data.append(d)

  for label in d['labels']:
    if label == CAT_INPUT_LABEL:
      num_cats += 1
    if label == DOG_INPUT_LABEL:
      num_dogs += 1

# Copy the cats/dogs into new array
images = np.empty((num_cats + num_dogs, 32, 32, 3), dtype=np.uint8)
labels = np.empty((num_cats + num_dogs), dtype=np.uint8)
index = 0

for data_batch in data:
  for batch_index, label in enumerate(data_batch['labels']):
    if label == CAT_INPUT_LABEL or label == DOG_INPUT_LABEL:
      # Data is stored in B x 3072 format, convert to B' x 32 x 32 x 3
      images[index, :, :, :] = np.transpose(
          np.reshape(data_batch['data'][batch_index, :],
          newshape=(3, 32, 32)),
          axes=(1, 2, 0))
      if label == CAT_INPUT_LABEL:
        labels[index] = CAT_OUTPUT_LABEL
      else:
        labels[index] = DOG_OUTPUT_LABEL
      index += 1

np.save('catdog_data.npy', {'images': images, 'labels': labels})

# Make sure images look correct
img  = Image.fromarray(images[10, :, :, :])
img.show()

