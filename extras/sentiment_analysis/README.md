# Sentiment Analysis Tutorial

This is the code and data used for the Sentiment Analysis Tutorial available
at: [INSERT LINK]()

In this tutorial we're going to show how to build a recurrent neural network
(RNN) that learns how to classify movie reviews as positive or negative using
TensorFlow high level APIs. Here, the focus is on introducing the high level APIs,
rather than on building a high-accuracy model.

## How to run this?

### Install TensorFlow

Go [here](https://www.tensorflow.org/install/) for instructions.
**Make sure you have installed TensorFlow v1.2 or higher**

### Train your model

The first time you run the script can take a while to the model actually
starts training, since it will first download the files available at
[LSTM Sentiment Analysis](https://github.com/adeshpande3/LSTM-Sentiment-Analysis)
where you can find a preprocessed version of the
[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
and also a
[great sentiment analysis tutorial using low-level TensorFlow by O’Reilly](https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow).

To start you can run the `sentiment_analysis.py` file. There are
a number of different arguments you can try in order to produce different models
so we encourage you to have a look on the code to see all you can do.

Below are some examples of what you can change in the model just using arguments.

```shell

# Train the model, then evaluate and predict using the default model,
# will save the output at "sentiment_analysis_output".
# The default model is a single LSTM layer with 128 cells
# and a dense softmax layer on top of the last cell.
# This was the \same command used to train the the model mentioned in
# the tutorial.
$ python sentiment_analysis.py

# Use the average of the hidden states as the final output for the RNN layers
# as suggested by "Sentiment Analysis with Deeply Learned Distributed
# Representations of Variable Length Texts" from James Hong and Michael Fang
# (2015) http://cs224d.stanford.edu/reports/HongJames.pdf.
# It will not run an experiment, instead it will use the estimator's
# interface to first train and then evaluate the model
# (this is the biggest difference of using/not using experiments in local
# settings).
$ python sentiment_analysis.py --use_hidden_states=average --dont_run_experiment

# Change the model to have 3 LSTM layers with dropout of 0.75, 0.5 and 0.5
# and add two DNN layers on top of the RNNs.
# It will train for 1 epoch
$ python sentiment_analysis.py --rnn_cell_sizes 128 128 64 \
                               --dropout_keep_probabilities 0.75 0.5 0.5 \
                               --dnn_layer_sizes 64 64 \
                               --model_dir "my_deep_rnn" \
                               --num_epochs 1

# There are more command line flags to play with; check sentiment_analysis.py
# for details.
```

### Using TensorBoard

Use this command during, or after training to visualize metrics,
the model graph, and more.
```shell
# Set --log_dir to point to the model_dir of the previous step
# by default the model_dir is "sentiment_analysis_output"
$ tensorboard --log_dir="sentiment_analysis_output"
```

![imgs/tensorboard](imgs/tensorboard.png)

You can also check your TensorFlow graph to debug your graph if needed.

![imgs/tensorboard_graph](imgs/tensorboard_graph.png)

### Training example output

```shell
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
INFO:tensorflow:global_step/sec: 4.19543
INFO:tensorflow:loss = 0.565088, step = 701 (23.835 sec)
INFO:tensorflow:global_step/sec: 3.94001
INFO:tensorflow:loss = 0.509592, step = 801 (25.381 sec)
INFO:tensorflow:global_step/sec: 4.19204
INFO:tensorflow:loss = 0.652886, step = 901 (23.855 sec)
INFO:tensorflow:global_step/sec: 4.06573
INFO:tensorflow:loss = 0.696719, step = 1001 (24.596 sec)
INFO:tensorflow:global_step/sec: 4.03502
INFO:tensorflow:loss = 0.519887, step = 1101 (24.783 sec)
INFO:tensorflow:global_step/sec: 3.93356
INFO:tensorflow:loss = 0.579439, step = 1201 (25.422 sec)
INFO:tensorflow:global_step/sec: 3.87702
```

### Exaluation example output

```shell
INFO:tensorflow:Starting evaluation at 2017-07-20-22:01:39
INFO:tensorflow:Restoring parameters from pre/model.ckpt-6262
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

### Prediction sample output

In the sentences below *unk* stands for a word that isn't in the word
embedding. The *0 0 ... 0* in some lines means that the line was padded
with zeros in the end so it could have length equals to 250 (more details
about this in the tutorial).

The model outputs a 2-dim tensor as result containing the probability
of a review be negative (index 0) or positive (index 1)

```shell
if this is the best commander hamilton movie i have no curiosity about the others a movie actors greatest tools are his eyes but when peter stormare wants to show great emotion he closes his so for five or six seconds we get to admire his eyelids while his feelings remain unknown behind them lousy acting technique stormare also flinches sometimes when he fires a gun turning his head away and clamping his eyes shut watch carefully james bond can rest easy with competition like this there are some interesting supporting performances from other actors but not enough to hang a whole movie on the cinematography is unk doing a fine job of capturing the nordic cold even the sahara winds up looking cold perhaps hamilton carries his own climate with him there are some individual good action sequences here unfortunately the only sense of humor on screen belongs to the villain which turns the hero into a big pill james bonds jokes may not be particularly good but at least he doesnt look constipated all the time one positive point in the movies favor is that the psychotic contorted vicious hatred of israel in unk books has been left out what has been kept in is worship of a noble heroic plo that he shows us functioning in libya without the dictator unk knowledge or supervision this fantasy is hard to believe since unk actually threw the plo out of libya for four years at a time and at
Prediction: [ 0.1009393   0.89906073]
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

## Done training? Play with your model!

Make sure you pass the same arguments to *sentiment_analysis.py*
and *sentiment_analysis.py --mode=classify* since in the classify mode
the script will load the model trained in the `model_dir` so make sure
you're running the same model.

**Only lower case letters and numbers are accepted as input.**

```shell
# The script will load your model and you can use it to classify new sentences
$ python sentiment_analysis.py --mode=classify --model_dir="sentiment_analysis_output"
```

### Output example

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
Write your review (or type <exit> to exit): it was not good
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.411701
Positive: 0.588299
Write your review (or type <exit> to exit): i thought the movie was incredible and inspiring
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.0676128
Positive: 0.932387
Write your review (or type <exit> to exit): this is a great movie
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.130054
Positive: 0.869946
Write your review (or type <exit> to exit): this is a good movie but isnt the best
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.224676
Positive: 0.775324
Write your review (or type <exit> to exit): this is a good movie
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.189163
Positive: 0.810837
Write your review (or type <exit> to exit): this is a good movie it is the best
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.0859528
Positive: 0.914047
Write your review (or type <exit> to exit): it was definitely bad
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.894781
Positive: 0.105219
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
Write your review (or type <exit> to exit): its not that bad i think its a good movie
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.427492
Positive: 0.572508
Write your review (or type <exit> to exit): its not bad i think its a good movie
Generating prediction...
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.489538
Positive: 0.510462
Write your review (or type <exit> to exit): its not good i think its a bad movie
INFO:tensorflow:Restoring parameters from pretrained_model/model.ckpt-4380
Negative: 0.538815
Positive: 0.461185
```

## Whats next?

In this tutorial we showed how to implement a Recurrent Neural Network for
binary sentiment analysis using TensorFlow high level APIs.

* We encourage you to run the code and see how the model performs for yourself.
  The model parameters were not tuned, so a good exercise is just play with
  the parameters and in order to have better results.
  Try changing the learning rate, optimizer, hidden state size,
  number of RNN cells, number of DNN layers, and so on.

* Finally, the model presented above can be easily changed to be used on
  different data and even perform different classification
  or prediction tasks. More details can be seen in the [code]().
  A great example is [colorbot](https://github.com/random-forests/tensorflow-workshop/blob/master/extras/colorbot/)
  a deep RNN model that receives a word (sequence of characters) as input
  and learns to predict a rgb value that better represents this word.
  As a result we have a color generator!
