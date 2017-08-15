# Sentiment Analysis Tutorial

### This is not updated: it was moved to a Doc, ask @monteirom

In this tutorial we're going to show how to build a Recurrent Neural
Network (RNN) that learns how to classify movie reviews as positive or
negative using TensorFlow high level APIs.

More specifically we're going to build a Recurrent Neural Network for a 
[Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
application using an [Estimator](https://www.tensorflow.org/extend/estimators).
Our task will consist of: given a sentence (sequence of words) classify
it as positive or negative.

The code and instructions about how to run it can be seen
[here #TODO: INSERT LINK]().

The goal is that at the end of this tutorial you'll be able to start
writing your own Estimator models and using TensorFlow APIs  through
this practical example and at the same time learn more about RNN models.
The model presented here doesn't get to the state of the art accuracy,
and is not our goal to do so, the dataset is actually too small for a
LSTM to have big advantages compared to simpler and faster methods, but
this is a great example of to get started with.
For a state of the art approach check
[this tutorial](https://www.tensorflow.org/tutorials/recurrent).

## Introduction

For this tutorial we understand you're already familiar with basic RNN
concepts and with TensorFlow. As a start point to Estimators we recommend
you to take a look at:
  * [Estimators](https://www.tensorflow.org/extend/estimators)

## Tutorial Files

This tutorial references the following files from [#TODO: INSERT LINK]():

File | Purpose
--- | ---
`sentiment_analysis.py` | The code that does training, evaluation and prediction.
`model_fn_lib.py` | The model function implementation.
`input_function_lib.py` | The input function implementation.
`imdb.py` | The code to read the dataset.
`sentiment_analysis_test.py` | Basic test to `model_fn_lib.py` and `input_function_lib.py`.

## About the Data

We'll use the
[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/),
which is a popular dataset for binary sentiment classification containing
25,000 reviews for training, and 25,000 reviews for testing with an
even number of positive and negative reviews.
```
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```

Examples of reviews in the dataset:

#### Negative Review

```
I was very curious to see this film, after having heard that it was clever
and witty. I had to stop halfway because of the unbearable boredom I felt.
The idea behind the film would have been acceptable: depicting the way
the relationship between a man and a woman evolves,
through all the problems and difficulties that two people living in a
big city can experience. What made me dislike the whole film were two things.
First of all, the film was so down-to-earth that it looked as if,
by describing the problems that a couple must solve on a day-to-day basis,
it became itself ordinary and dull.
Secondly, the overall sloppiness of the production,
with dialogues that were barely understandable. Too bad.
```

#### Positive Review

```
Cinematically, this film stinks. So does a lot of the acting.
But I don't care. If there is a strong representation of what the 80's
were like(For a lot of us in the innercity anyways) and what hip-hop,
Zulu nation, and break dancing were really like.Great music, great dancing!
It almost seems like a documentary of a time now past when hip hop was a way
of life. It's also interesting to see New York looking like ground zero
from a nuclear attack. Some viewers may be too young to remember that It was
a poor, run down city during the 70's and 80's. This is the best of all
the hip-hop/break dancing movies that came out around that period.
Of course the 80's are considered a joke now with all the bad tv shows and
movies, but those of us who lived through it will always remember it fondly
for a time when music, dancing, and graffiti were fresh, yo!
```

### Prepare the Data

When dealing with 
[NLP](https://en.wikipedia.org/wiki/Natural_language_processing)
tasks it's very important to preprocess your data and to choose a good
way to represent it, we're not going into details about how doing it in
the best way possible, since this could be another complete tutorial,
instead we're just going to describe how we did for this particular
problem.

#### Preprocess the Data

In this example the reviews were padded with zeros or truncated to have
length equal 250. This was done mainly in order to avoid very long sequences.

Padding is a common practice when working with RNNs but is not mandatory when
working with RNNs on TensorFlow and is not the most efficient approach to pad
all the sequences to the same length, we're going into more details about
more efficient approaches later in this tutorial .

All characters were converted to lowercase and punctuation was removed
for simplicity.

> Note: punctuation and special characters can say a lot about emotions,
  in order to have more expressive results maybe keeping and treating those
  correctly can be a good idea.

#### Numeric Representation

Neural Networks expect numeric inputs, this means we need to represent
text as a numeric value. There are many possible approaches, two classical
ways to do it are to:
  * Segment the text into words, representing each word as a vector;
  * Segment the text into characters, representing each character as a vector.

Once you segmented the text, another question is how to represent the words
or characters as vectors? There are also two popular ways to do so:
  * [One hot representation](https://www.tensorflow.org/api_docs/python/tf/one_hot):
  sparse and high-dimensional vectors (not so efficient when dealing with
  sequences in a word level, but can be really useful when dealing
  with sequences in a character level);
  * [Word Embedding](https://www.tensorflow.org/tutorials/word2vec):
  also called word vectors, dense and low-dimensional vectors.
  You can train your own embedding along with your model, or you can
  use a pre-trained embedding.

In this tutorial we'll segment the reviews into words and use a pre-trained
word embedding to convert the words to a vector representation. This word
embedding was trained using the
[GloVe algorithm](https://nlp.stanford.edu/projects/glove/)
and contains 400000 words as 50 dim vectors. As a result we have a matrix
of shape [400000, 50] where each row is a word representation.

> Note: Thanks to [@adeshpande3](https://github.com/adeshpande3/LSTM-Sentiment-Analysis)
  for providing this word embedding!

![](imgs/embedding.png)

Each word in the review will be converted to an index that points to a row in the
embedding. Each row has a 50 dim vector that better represents a particular
word.

## About the Model

Our model will consist of a
[LSTM cell](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/rnn/LSTMCell)
with a dense softmax layer on top of it. The final output is the probability of a
review to be a positive (index 1) or negative review (index 0).

![](imgs/model.png)

Before going into more details about the model itself, let's discuss
briefly what is needed in order to implement an Estimator on TensorFlow.

### Estimators

Estimators are a high Level abstraction that support all the basic
operations you need on a Machine Learning model. They encode best
practices, are ready for deployment with tensorflow/serving and
are distributed and scalable by design.

![](imgs/estimator.png)

In order to implement our own Estimator we basically need:
  * An [input function](https://www.tensorflow.org/get_started/input_fn):
    this is the input pipeline implementation, where you're going to
    process your data and return the features and labels that will be used
    for training, evaluation and prediction using the Estimator interface.
  * A [model function](https://www.tensorflow.org/extend/estimators#constructing_the_model_fn):
    where will actually define our model, and the training, evaluation and
    prediction operations.

For more about how to implement an Estimator check
[this tutorial](https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/Estimator)
or watch this talk
[TensorFlow High level APIs at TensorFlow Dev Summit 2017](https://www.youtube.com/watch?v=t64ortpgS-E).

Now let's have a look at the code!

### Input function

If your data fits in memory (and you're okay about loading it in memory)
there are "prebuilt" input functions for
[numpy](https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/inputs/numpy_input_fn)
and [pandas](https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/inputs/pandas_input_fn).

But if you need to manipulate the data with more complex operations or if
the data doesn't fit in memory, an efficient and scalable way to implement
your own input function is to use the
[Dataset API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/programmers_guide/datasets.md).

The Dataset API enables you to build complex input pipelines from simple,
reusable pieces, making it easy to deal with large amounts of data, different
data formats, and complicated transformations.

Here's an input function implementation using the Dataset API.

```python
def get_input_fn(x_in, y_in, batch_size,
                 shuffle=True, epochs=1,
                 batch_by_seq_len=False,
                 max_length=250):
  """Returns an input function used by train_sentiment_analysis.py."""
  def input_fn():
    """Input function."""
    # calculates the length of the sequences
    # since the inputs are already padded with zeros in the end
    # the length will be the last index non zero + 1
    x_len = np.array(
        [np.nonzero(seq)[0][-1] + 1 for seq in x_in]).astype('int32')

    # creates the dataset from in memory data
    # x_in: sequence of indexes that map a word to an embedding
    # x_len: sequence lengths
    # y_in: 1 if positive review, 0 if negative review
    ds = tf.contrib.data.Dataset.from_tensor_slices((x_in, x_len, y_in))

    # repeats the dataset `epochs` times.
    ds = ds.repeat(epochs)

    if shuffle:
      # make sure the buffer is big enough for your data
      ds = ds.shuffle(buffer_size=25000 * 2)

    # batches the data
    ds = ds.batch(batch_size)

    # creates iterator
    x, x_len, y = ds.make_one_shot_iterator().get_next()

    # feature must be a dictionary
    dict_x = {'x': x, rnn_common.RNNKeys.SEQUENCE_LENGTH_KEY: x_len}
    return dict_x, y

  return input_fn
```

The Dataset API introduces two new abstractions to TensorFlow: **datasets**
and **iterators**.

* A Dataset can either be a source or a transformation:
  * Creating a source (e.g. Dataset.from_tensor_slices()) constructs a dataset
    from one or more tf.Tensor objects.
  * Applying a transformation constructs a dataset from one or more
    tf.contrib.data.Dataset objects.
    * Repeat: produce multiple epochs;
    * Shuffle: it maintains a fixed-size buffer and chooses the next element
      uniformly at random from that buffer;
    * Batch:  constructs a dataset by stacking consecutive elements of another
      dataset into a single element.

* A Iterator provides the main way to extract elements from a dataset.
  The Iterator.get_next() operation yields the next element of a Dataset, and
  typically acts as the interface between input pipeline code and your model.

The implementation above is not the most efficient way to batch the data for
a RNN, we're wasting time and space by padding all the batches to have length
equals to 250.

![](imgs/batch.png)

One possible approach to make it more efficient is to put sequences with
similar length in the same batch. We implement this in the code below.

```python
def get_input_fn(x_in, y_in, batch_size,
                 shuffle=True, epochs=1,
                 batch_by_seq_len=False,
                 max_length=250):
  """Returns an input function used by train_sentiment_analysis.py."""
  def _length_bin(length, max_seq_len, length_step=10):
    """Sets the sequence length bin."""
    bin_id = (length // length_step + 1) * length_step
    return tf.cast(tf.minimum(bin_id, max_seq_len), tf.int64)

  def _make_batch(key, ds):
    """Removes extra padding and batches the bin."""
    # eliminate the extra padding
    key = tf.cast(key, tf.int32)
    ds = ds.map(lambda x, x_len, y: (x[:key], x_len, y))

    # convert the entire contents of the bin to a batch
    ds = ds.batch(batch_size)
    return ds

  def input_fn():
    """Input function."""
    # calculates the length of the sequences
    # since the inputs are already padded with zeros in the end
    # the length will be the last index non zero + 1
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
      # manually implement bucket by sequence length
      # the idea is to make batches with sequences of similar length
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
```

We're using a transformation called `group_by_window` that maps each
consecutive element in this dataset to a key using a `key_func` and then
groups the elements by key. It then applies `reduce_func` to at most
`window_size` elements matching the same key.

We're using the `group_by_window` transformation to batch reviews that
have similar length together, since the batches are created based on the
sequence length the reviews in the same batch will have approximately the
same length which means we'll have less padding saving space and computation
time.

Using this more complex implementation, that we called `batch_by_seq_len`,
on this specific dataset I can see an improvement of 2 global_step/sec
running it on my local machine, in other words if we take ~16 seconds to
process 100 batches using the usual batch implementation, now we take ~12
seconds to process 100 batches.

![](imgs/batch_by_seq_len.png)

For more details about padding and batching with RNNs watch
this great talk:
[Sequence Models and the RNN API (TensorFlow Dev Summit 2017)](https://youtu.be/RIR_-Xlbp7s?t=4m14s)

You can see the all the input function implementations used in this
tutorial at [#TODO: INSERT LINK]().

We can create different input functions calling `get_input_fn`, like:

```python
# input functions
train_input_fn = get_input_fn(x_train, y_train, FLAGS.train_batch_size,
                              epochs=FLAGS.num_epochs,
                              batch_by_seq_len=not(FLAGS.dont_batch_by_seq_len))

eval_input_fn = get_input_fn(x_eval, y_eval, FLAGS.eval_batch_size, epochs=1)

sample_input_fn = get_input_fn(x_sample, y_sample, 1, epochs=1, shuffle=False)
```

### Model Definition

We'll define our model implementing a model function, where we'll also define
the operations used for training, evaluation and prediction. In this tutorial
we'll focus on the model itself, and we'll comment briefly about the
operations chosen, since you can easily learn more about the operations and
what they're doing in the TensorFlow documentation and other online materials.

Our model function definition looks like:

```python
def model_fn(features, labels, mode, params):
  # model and operations definition
  ...
  # estimator definition
  return EstimatorSpec(...)
```

Where the `features` and `labels` are returned by the input function we just
defined, the `mode` is a string value indicating the context in which the
`model_fn` was invoked (TRAIN, EVAL, PREDICT) and `params` is an optional
argument containing a dict of hyperparameters used for training. More details
[here](https://www.tensorflow.org/extend/estimators#constructing_the_model_fn).

The complete model implementation can be found at
[#TODO: INSERT LINK]().

Let's have look at the model function implementation.

#### Embedding

First, we need to represent the words as vectors, the `features['x']`
is a tensor with indexes mapping the word to a row in the `word_vector`
matrix (the pre-trained word embedding).

This is simplest code to load the embedding and convert the indexes to
vectors, there's a discussion about how to do this in a more efficient way
[here](https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow).

```python
  # get the sequences from the features dict
  review = features['x']

  # defining constant to store pre-trained embedding
  W = tf.constant(word_vector, name='W')
  # indexes -> vectors using the embedding
  data = tf.nn.embedding_lookup(W, review)
```

Once we converted the indexes to actual vectors, the `data` variable defined
above will be a 3-dim vector with shape [BATCH_SIZE, MAX_LENGTH, 50]

![](imgs/input.png)


#### RNN

Now that we have our input in the expected format we can implement the model
itself.

Our model consists of a Recurrent Neural Network (RNN). Neural networks
like densely-connected networks and Convolutional Neural Networks
have no memory, which means that each input is processed independently.
RNNs are neural networks that have memory, in other words they have
an internal state that is updated based on the seen inputs and on
it's own previous state (memory).

![](imgs/RNN.png)
Image from: http://colah.github.io/posts/2015-08-Understanding-LSTMs/  

Another way to see the RNN, is just unroll it over time. This way a RNN can
be seen as multiple copies (cells) of the same network, where each copy shares
information about what it has seen to the next cell.

![](imgs/RNN_unroll.png)
Image from: http://colah.github.io/posts/2015-08-Understanding-LSTMs/  

The Pseudo-code to run and update the state of a basic RNN cell
would be something similar to:

```python
for t in range(len(inputs)):
  # updates internal state
  hidden_state = activation_fn(W_x * inputs[t] + W_h * hidden_state + bias)
```

To learn more about RNNs check:
 * [Understanding LSTM Networks at colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
 * [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

In this tutorial we will actually implement a special type of RNN cell called
Long Sort Term Memory Cell (LSTM), which is capable of learning
long-term dependencies.

TensorFlow allow us to implement complex cell types and operations in a few
lines of code!

In the code below we're creating multiple LSTMCells, then adding dropout
in the output and hidden state of each of them if running in training mode.

```python
if dropout_keep_probabilities:
  # if we're not training we want to keep all RNN cells
  if is_training:
    probabilities = [1] * len(dropout_keep_probabilities)
  else:
    probabilities = dropout_keep_probabilities

  rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(
      tf.nn.rnn_cell.LSTMCell(size),
      output_keep_prob=keep_prob,
      state_keep_prob=keep_prob)
                for size, keep_prob in zip(rnn_cell_sizes,
                                           probabilities)]
```

Once we created the cells, we can stack them.

```python
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
```

All the RNN code above just created the RNN cells (you can see all the RNN
cell types available
[here](https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/rnn_cell))
in order to "unroll" the cells we can use
[tf.nn.dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
that will actually implement the RNN "for loop", returning all the outputs
over time and the final state.

```python
# outputs: [BATCH_SIZE, SEQUENCE_LENGTH, STATE_SIZE]
# final state: tuple where the first element is a tf.contrib.rnn.LSTMStateTuple
outputs, final_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                         inputs=data,
                                         dtype=tf.float32)
```

On top of the RNN we can add other neural network layers, for this example
we'll add a dense layer, in order to stack these two layers we need to define
what will be the final output from the RNN layers.

In this implementation we can get the output from the last step (usual
implementation) or get the average from the hidden states as suggested by
["Sentiment Analysis with Deeply Learned Distributed Representations of Variable Length Texts" from James Hong and Michael Fang (2015)](http://cs224d.stanford.edu/reports/HongJames.pdf).

```python
if average_hidden_states:
  last_activations = tf.div(
      tf.reduce_sum(outputs, axis=1),
      tf.cast(sequence_length[:, tf.newaxis], dtype=tf.float32))
else:
  # slice to keep only the last cell of the RNN
  last_activations = rnn_common.select_last_activations(outputs,
                                                        sequence_length)
```

#### Dense Softmax Layer

Adding dense layers to the model is very straight forward with the
[tf.layers API](https://www.tensorflow.org/api_docs/python/tf/layers).

```python
# final dense layer for prediction
predictions = tf.layers.dense(last_activations, label_dimension)
predictions_softmax = tf.nn.softmax(predictions)
```

#### Defining operations

After defining our model we can just specify which operations to run on each
execution mode. We are using accuracy as evaluation metric, calculating
the loss using Softmax Cross Entropy, specifying how to optimize the loss,
and defining the predict operation.

```python
# define model operations
loss = None
train_op = None
eval_op = None

if not is_predict:
  labels_onehot = tf.one_hot(labels, 2)
  loss = tf.losses.softmax_cross_entropy(labels_onehot, predictions)

if is_eval:
  eval_op = {
      'accuracy': tf.metrics.accuracy(
          tf.argmax(input=predictions_softmax, axis=1),
          tf.argmax(input=labels_onehot, axis=1))
  }

if is_training:
  train_op = tf.contrib.layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer=optimizer,
      learning_rate=learning_rate)

# return the estimator definition
return tf.estimator.EstimatorSpec(mode,
                                  predictions=predictions_softmax,
                                  loss=loss,
                                  train_op=train_op,
                                  eval_metric_ops=eval_op)
```

## Training

Now we can just create the Estimator using the model function above,
and call the methods available on the Estimator interface.

```python
estimator.train(input_fn=train_input_fn)
```

```
INFO:tensorflow:loss = 0.691962, step = 101 (36.537 sec)
INFO:tensorflow:global_step/sec: 3.79198
INFO:tensorflow:loss = 0.637554, step = 201 (26.371 sec)
INFO:tensorflow:global_step/sec: 4.12
INFO:tensorflow:loss = 0.461921, step = 301 (24.272 sec)
INFO:tensorflow:global_step/sec: 4.23288
INFO:tensorflow:loss = 0.456651, step = 401 (23.625 sec)
INFO:tensorflow:global_step/sec: 4.18946
INFO:tensorflow:loss = 0.603483, step = 501 (23.869 sec)
INFO:tensorflow:global_step/sec: 4.07666
INFO:tensorflow:loss = 0.617782, step = 601 (24.530 sec)
....
INFO:tensorflow:loss = 0.696719, step = 1001 (24.596 sec)
INFO:tensorflow:global_step/sec: 4.03502
INFO:tensorflow:loss = 0.519887, step = 1101 (24.783 sec)
INFO:tensorflow:global_step/sec: 3.93356
INFO:tensorflow:loss = 0.579439, step = 1201 (25.422 sec)
INFO:tensorflow:global_step/sec: 3.87702
```

## Evaluation

```python
estimator.evaluate(input_fn=eval_input_fn)
```

```
INFO:tensorflow:Evaluation [1/100]
INFO:tensorflow:Evaluation [2/100]
INFO:tensorflow:Evaluation [3/100]
INFO:tensorflow:Evaluation [4/100]
INFO:tensorflow:Evaluation [5/100]
INFO:tensorflow:Evaluation [6/100]
INFO:tensorflow:Evaluation [7/100]
INFO:tensorflow:Evaluation [8/100]
INFO:tensorflow:Evaluation [9/100]
INFO:tensorflow:Evaluation [10/100]
...
INFO:tensorflow:Evaluation [100/100]
INFO:tensorflow:Finished evaluation at 2017-07-24-20:39:32
INFO:tensorflow:Saving dict for global step 6262: accuracy = 0.856875, global_step = 6262, loss = 0.374715
```

## Training and Evaluation in a Distributed Environment

As mentioned before a great thing about Estimators is that they are
distributed and scalable by design. In order to run the model in a distributed
way using data-parallelism you just need to create an
[Experiment](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment).
Experiments know how to invoke train and eval in a sensible fashion for
distributed training.

Below is all the code used to create and run an Experiment.

```python
def get_experiment(estimator, train_input, eval_input):
  def _experiment_fn(run_config, hparams):
    """Creates experiment.

    Experiments perform training on several workers in parallel,
    in other words Experiments know how to invoke train and eval
    in a sensible fashion for distributed training.

    We first prepare an estimator, and bundle it
    together with input functions for training and evaluation
    then collect all that in an Experiment object.
    """
    del run_config, hparams  #unused args
    return tf.contrib.learn.Experiment(
        estimator,
        train_input_fn=train_input,
        eval_input_fn=eval_input
    )
  return _experiment_fn

# run training and evaluation using an Experiment
learn_runner.run(get_experiment(estimator, train_input, eval_input),
                 run_config=run_config)

```

## Predicting

Once we're done training we can check how well the model classify new reviews.

In this case we're just classifying 5 new sentences randomly chosen from the
eval dataset. In the sentences below *unk* stands for a word that isn't
represented in the word embedding.

```python
predictions = estimator.predict(input_fn=sample_input)
```

```
if this is the best commander hamilton movie i have no curiosity about the others a movie actors greatest tools are his eyes but when peter stormare wants to show great emotion he closes his so for five or six seconds we get to admire his eyelids while his feelings remain unknown behind them lousy acting technique stormare also flinches sometimes when he fires a gun turning his head away and clamping his eyes shut watch carefully james bond can rest easy with competition like this there are some interesting supporting performances from other actors but not enough to hang a whole movie on the cinematography is unk doing a fine job of capturing the nordic cold even the sahara winds up looking cold perhaps hamilton carries his own climate with him there are some individual good action sequences here unfortunately the only sense of humor on screen belongs to the villain which turns the hero into a big pill james bonds jokes may not be particularly good but at least he doesnt look constipated all the time one positive point in the movies favor is that the psychotic contorted vicious hatred of israel in unk books has been left out what has been kept in is worship of a noble heroic plo that he shows us functioning in libya without the dictator unk knowledge or supervision this fantasy is hard to believe since unk actually threw the plo out of libya for four years at a time and at
Prediction: [ 0.1009393  0.89906073]
Label: 0
this is the most stupid movie ever made the story is laughable his wife and kid think hes insane then they dont then it turns out he is and i think they knew it all along there is a dog named ned that causes some problems and i think its all his unk does jim carey god only knows why virginia madsen took this unk is a career sinker i think the target audience for this is 11 and 12 year olds and that adds up to 23 or maybe its for 8 and 10 years olds which also adds up to 23 or maybe its for really dumb 23 year olds or maybe really dumb 32 year olds because thats 23 in reverse or maybe 46 year olds would enjoy it because half of that is 23 i think looking up things on the internet about the number 23 would be more entertaining than this movie unless you wanted to see a comedy 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Prediction: [ 0.74981982  0.25018018]
Label: 0
there isnt much that comes close to the unk storytelling and suspenseful unk levels as goldeneye when it came out it was the greatest game of alltime and even today it stays strong i will admit that this game did get boring after a few months of playing and by not playing it again until two years later i was thrust back into its greatest almost as if i was playing it for the first time again there are 20 unk levels which is probably the most of any james bond game to date probably the most unforgettable one is the tank level which was likely the most explosive video game sequence at that time and the unk shooting as well as usage of q gadgets is what james bond fans are always dying to use frankly as a james bond fan i look for aspects of a true james bond experience which are now showing up in the ps2 games so this game while it has some great action and usable gadgets i was somewhat expecting a little more even back in 1997 i also disliked that this game didnt have q or m or moneypenny or anyone from mi6 while watching the movies bond interacts with these characters at least a few times throughout each movie but they are nowhere to be seen in this game and vocal dialogue would have made the game more lively rather than the text dialogue they wound up using they had the
Prediction: [ 0.12032966  0.87967038]
Label: 1
as a true canadian i always avoid canadian movies however now and then i get trapped into watching one this one is better than most which is to say mediocre it has many of the usual flaws of canadian unk unk excess of cinematic gimmicks and above all the unk canadian habit of using canadian cities as unk for american ones i mean using the historic metropolis of montreal as a stand in for harrisburg pennsylvania is just short of obscene i was in a generous mood i gave it a 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Prediction: [ 0.71495974  0.28504029]
Label: 0
if youre interested in learning about the real side of spying this movie is for you unlike 007 movies this shows how things really go down in the world of espionage timothy hutton and sean penn both give outstanding performances in this unk film certainly worth watching 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Prediction: [ 0.25497162  0.74502844]
Label: 1
```

### Classifying New Data

You can also try to classify new sentences with this model.

```shell
# The script will load your model and you can use it to classify new sentences
# by default the model_dir is "sentiment_analysis_output"
$ python sentiment_analysis.py --mode=classify --model_dir="sentiment_analysis_output"
```

Make sure you pass the same arguments to `sentiment_analysis.py` and
`sentiment_analysis.py --mode=classify`, since it will load the same model you
just trained make sure you're building the same model running in both modes.

Here are some sentences we tried with a model that got 82% accuracy, we can get
about 86% accuracy with the model training it for ~6000 steps, but we found that
the model trained around ~4000 is more fun to play with.

```
Write your review (or type <exit> to exit): it was fine i guess
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.527832
Positive: 0.472169
Write your review (or type <exit> to exit): it was good
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.357005
Positive: 0.642995
Write your review (or type <exit> to exit): it wasnt good
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.60162
Positive: 0.39838
Write your review (or type <exit> to exit): this is a great movie
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.130054
Positive: 0.869946
Write your review (or type <exit> to exit): its not that bad
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.740116
Positive: 0.259884
Write your review (or type <exit> to exit): it is bad
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.906015
Positive: 0.0939852
Write your review (or type <exit> to exit): its not bad i think its a good
movie
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.489538
Positive: 0.510462
Write your review (or type <exit> to exit): its not good i think its a bad
movie
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.538815
Positive: 0.461185
```

We can see that the model learned some interesting relations,
but is definitely not perfect and can be improved.

## Visualizing your Model

When using estimators you can also visualize your data in
[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard),
with no changes in your code. You can use TensorBoard to visualize your TensorFlow graph,
plot quantitative metrics about the execution of your graph,
and show additional data like images that pass through it.

Here's what you see if you run TensorBoard in the `model_dir` you used for your model.

```shell
# Check TensorBoard during training or after it.
# Just point TensorBoard to the model_dir you chose on the previous step
# by default the model_dir is "sentiment_analysis_output"
$ tensorboard --log_dir="sentiment_analysis_output"
```

![imgs/tensorboard](imgs/tensorboard.png)

You can also visualize your TensorFlow graph, which is very useful for debugging purposes.

![imgs/tensorboard_graph](imgs/tensorboard_graph.png)

## What's next?

In this tutorial we showed how to implement a recurrent neural network for
binary sentiment analysis using TensorFlow high level APIs.

* We encourage you to run the code and see how the model performs for yourself.
  The model parameters were not tuned, so a good exercise is just play with
  the parameters and in order to have better results.
  Try changing the learning rate, optimizer, hidden state size,
  number of RNN cells, number of DNN layers, and so on.

* Finally, the model presented above can be easily changed to be used on
  different data and even perform different classification or prediction tasks.
  More details can be seen in the [code #TODO: INSERT LINK]().
  A great example is
  [colorbot](https://github.com/mari-linhares/tensorflow-workshop/blob/master/code_samples/RNN/colorbot/)
  a deep RNN model that receives a word (sequence of characters) as input and
  learns to predict a rgb value that better represents this word. As a result
  we have a color generator!

* Learn more about:
  * [RNNs](https://www.tensorflow.org/tutorials/recurrent)
  * [Estimators](https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/Estimator)
  * [Dataset API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/programmers_guide/datasets.md)
  * [Word Embeddings](https://www.tensorflow.org/tutorials/word2vec)

