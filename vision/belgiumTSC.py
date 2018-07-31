"""Builds the network.
Implements the inference/loss/training pattern for model building.
1. inference() - Builds the model as far as required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply optimization.
This file is used by the various files and not meant to be run.
"""

import tensorflow as tf

# Code Structure Inspired by:
## https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py


# The BelgiumTSC dataset has 62 classes, representing the digits 0 through 61.
NUM_CLASSES = 62

# The mean dimensions of BelgiumTSC images are around 128x128 pixels, with different aspect ratio
# Resizing to 32x32 works decently, hence this represents resized size
IMAGE_SIZE = 32

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3

# IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS

def inference(images_ph):
	"""Build the model up to where it may be used for inference.
	Args:
		images_ph: Images placeholder
	Returns:
		Output tensor with the computed logits.
	"""
	# Flatten input from: [None, height, width, channels]
	# To: [None, height * width * channels] == [None, 3072]
	images_flat = tf.contrib.layers.flatten(images_ph)
	# Fully connected layer. 
	# Generates logits of size [None, 62]
	images_flat = tf.contrib.layers.flatten(images_ph)
	logits = tf.contrib.layers.fully_connected(images_flat, NUM_CLASSES, tf.nn.relu)
	return logits

def loss(logits, labels_ph):
	"""Calculates the loss from the logits and the labels.
		Args:
		logits: Logits tensor, float - [batch_size, NUM_CLASSES].
		labels_ph: Labels tensor, int32 - [batch_size].
		Returns:
		loss: Loss tensor of type float.
	"""
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))
	return loss

def training(loss, learning_rate=0.001):
	"""Sets up the training Ops.
		Creates a summarizer to track the loss over time in TensorBoard.

		Creates an optimizer.

		The Op returned by this function is what must be passed to the
		`session.run()` call to cause the model to train.

		Args:
			loss: Loss tensor, from loss().
			learning_rate: The learning rate to use for optimizer.

		Returns:
			train_op: The Op for training.
	"""

	# Create the optimizer with the given learning rate.

	# GradientDescentOptimizer
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate)

	# AdamOptimizer
	optimizer = tf.train.AdamOptimizer(learning_rate)

	## Create a variable to track the global step.
	# global_step = tf.Variable(0, name='global_step', trainable=False)

	# Use the optimizer to apply the optimization that minimize the loss
	train_op = optimizer.minimize(loss)

	# (and also increment the global step counter) as a single training step.
	# train_op = optimizer.minimize(loss, global_step=global_step)

	return train_op

def evaluation(logits):	
	"""Evaluate the quality of the logits at predicting the label.
	Args:
		logits: Logits tensor, float - [batch_size, NUM_CLASSES].
	Returns:
		A scalar int32 tensor with the number of examples that were predicted correctly.
	"""
	# In this application, we just need the index of the largest value, which corresponds to the id of the label
	# Convert logits to label indexes.
	# Shape [None], which is a 1D vector of length == batch_size.
	# The argmax output will be integers in the range 0 to 61.
	predicted_labels = tf.argmax(logits, 1)
	return predicted_labels
