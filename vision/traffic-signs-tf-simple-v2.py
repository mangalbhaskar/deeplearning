import argparse
import os
import random
import skimage
import skimage.transform
import skimage.data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

# custom imports
import belgiumTSC


# Defaults
DEFAULT_ROOT_PATH = "./"

FLAGS = {}
FLAGS['DATA_ROOT'] = "data"
FLAGS['PROBLEM'] = 'traffic-signs'
FLAGS['DATASET'] = 'BelgiumTSC'
FLAGS['LOG_DIR'] = 'nnmodels/'+FLAGS['PROBLEM']+'/'+FLAGS['DATASET']
FLAGS['ARCH'] = 'simple'
FLAGS['MODEL_FILE'] = FLAGS['PROBLEM']+'-tf-model-'+FLAGS['ARCH']
# Basic model parameters as external flags.

FLAGS['learning_rate'] = 0.001
FLAGS['summary_at'] = 10

# epocs or max_steps is same meaning
# FLAGS['n_epochs'] = 458
FLAGS['n_epochs'] = 21

# Create a graph to hold the model
# graph = tf.Graph()

# with tf.Session as session:
# 	# saver = tf.train.import_meta_graph('./nnmodels/traffic-signs-tf-model-simple.meta')
# 	# saver.restore(session, tf.train.latest_checkpoint('./nnmodels'))
# 	saver = tf.train.Saver()
# 	saver.restore(session, './nnmodels/traffic-signs-tf-model-simple')


def load_data(data_dir):
  # Get all subdirectories of data_dir. Each represents a label.
  directories = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]
  # Loop through the label directories and collect the data in
  # two lists, labels and images.
  labels = []
  images = []
  for d in directories:
    label_dir = os.path.join(data_dir, d)
    file_names = [os.path.join(label_dir, f) 
                  for f in os.listdir(label_dir) 
                  if f.endswith(".ppm")]
    for f in file_names:
      images.append(skimage.data.imread(f))
      labels.append(int(d))
  return images, labels


def display_label_images(images, label):
	# Display images of a specific label
		limit = 24 # show max of 24 images
		plt.figure(figsize=(15,5))
		i = 1
		start = labels.index(label)
		end = start + labels.count(label)
		for image in images[start:end][:limit]:
			plt.subplot(3,8,i)
			plt.axis('off')
			i += 1
			plt.imshow(image)
		plt.show()


def display_images_and_labels(images, labels):
	# Display the first image of each label
	unique_labels = set(labels)
	plt.figure(figsize=(15,15))
	i=1
	# ar = []
	w = []
	h = []
	info = []
	for label in unique_labels:
		# pick the first image for each label
		image = images[labels.index(label)]
		plt.subplot(8,8,i)
		plt.axis("off")
		plt.title("{0} ({1})".format(label,labels.count(label)))
		i += 1
		_ = plt.imshow(image)
		plt.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0.05,hspace=0.05)

		x = image.shape[:2]
		info.append([image.min(), image.max()]) # TBD: raise a flag when values are not between 0-255 range
		w.append(float(x[0]))
		h.append(float(x[1]))
		# ratio = float(x[0])/float(x[1])
		# ar.append(ratio)
	# plt.tight_layout()
	# plt.savefig("TraficcSign-belgiumTS.png",dpi=100)
	# ar=np.asarray(ar)

	print("\nDimensions:\n\tWidth=> Min:{0}; Max:{1}; Median:{2}; Mean:{3} \n\tHeight=> Min:{4}; Max:{5}; Median:{6}; Mean:{7}".format( np.min(w), np.max(w), np.median(w), np.mean(w), np.min(h), np.max(h), np.median(h), np.mean(h) ))
	ar = np.asarray(w)/np.asarray(h)
	print("\nAspect Ratio Stats (Width/Height):\nMin: {0}; Max: {1}; Median: {2}; Mean: {3}; STD: {4}".format(np.min(ar), np.max(ar), np.median(ar), np.mean(ar), np.std(ar) ))
	print("image array:[min,max]: {0}".format(info))
	plt.show()


def placeholder_inputs():
	# Placeholedrs for inputs and labels
	# tf.placeholder(tf.float32, [None, height, width, channels]) 
	# [batch_size, height, width, channels] or NHWC
	# The None for batch size means that the batch size is flexible, which means that we can feed different batch sizes to the model without having to change the code. 	
	batch_size = None
	images_ph = tf.placeholder(tf.float32, [batch_size, belgiumTSC.IMAGE_SIZE, belgiumTSC.IMAGE_SIZE, belgiumTSC.IMAGE_CHANNELS], name="ground_truth_images")
	labels_ph = tf.placeholder(tf.int32, [batch_size], name="ground_truth_labels")
	return images_ph, labels_ph

def run_training(images32,labels):
	# Get the sets of images and labels for training, validation, and testing
	images_a, labels_a = np.array(images32), np.array(labels)
	print("labels: ", labels_a.shape, "\nimages:", images_a.shape)

	# Create a graph to hold the model
	graph = tf.Graph()
	# Create a model in the graph
	# This is so they become part of my graph object rather than the global graph.
	with graph.as_default():
		
		# Generate placeholders for the images and labels.
		images_ph, labels_ph = placeholder_inputs()		

		# Build a Graph that computes predictions from the inference model.
		logits = belgiumTSC.inference(images_ph)
		
		# Add to the Graph the Ops for loss calculation.
		loss = belgiumTSC.loss(logits, labels_ph)

		# Add to the Graph the Ops that calculate and apply optimization.
		train_op = belgiumTSC.training(loss, FLAGS['learning_rate'])

		# Add the Op to compare the logits to the labels during evaluation.
		predicted_labels = belgiumTSC.evaluation(logits)

		# Build the summary Tensor based on the TF collection of Summaries.
		summary = tf.summary.merge_all()

		# Add the variable initializer Op.
		# init = tf.initialize_all_variables() # is deprecated and will be removed after 2017-03-02
		init = tf.global_variables_initializer()

		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver()

		# Create a session for running Ops on the Graph.
		session = tf.Session(graph=graph)

		## Instantiate a SummaryWriter to output summaries and the Graph.
		summary_writer = tf.summary.FileWriter(DEFAULT_ROOT_PATH+FLAGS['LOG_DIR'], session.graph)

		# And then after everything is built:

		# Run the Op to initialize the variables.
		session.run([init])

 		# Start the training loop.
		for i in range(FLAGS['n_epochs']):
			start_time = time.time()
			feed_dict = {images_ph:images_a, labels_ph:labels_a}

			# Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to session.run() and the value tensors will be
      # returned in the tuple from the call.
			_, loss_value = session.run(
				[train_op, loss],
				feed_dict=feed_dict
		 	)

		 	duration = time.time() - start_time

			# Write the summaries and print an overview fairly often.
			if i % FLAGS['summary_at'] == 0:
				# Print status to stdout.
				print("Loss:[{}]:{} [{} sec]".format(i, loss_value, duration))
				# Update the events file.
				# summary_str = session.run(summary, feed_dict=feed_dict)
				# summary_writer.add_summary(summary_str, i)
				# summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
			if ((i + 1) % 1000 == 0 or (i + 1)) == FLAGS['n_epochs']:
				# checkpoint_file = os.path.join(DEFAULT_ROOT_PATH+FLAGS['LOG_DIR'], 'model.ckpt')
				checkpoint_file = DEFAULT_ROOT_PATH+FLAGS['LOG_DIR']+'/'+FLAGS['MODEL_FILE']
				saver.save(session, checkpoint_file)
				# Evaluate against the training set.
				print('Training Data Eval:')
				# For Quick Analysis, pick 10 random images
				sample_indexes = random.sample(range(len(images32)),10)
				sample_images = [images32[i] for i in sample_indexes]
				sample_labels = [labels[i] for i in sample_indexes]
				print(sample_labels)
				
				# Run the predicted_lables op.
				sample_predicted = do_eval(session,
					predicted_labels,
					images_ph,
					sample_images)

				print(sample_predicted)
				
				# Visualize the predictions and the ground truth
				do_eval_viz(sample_images,
					sample_labels,
					sample_predicted)

				# Evaluate against the validation set.
				print('Validation Data Eval: -- Not Present')
        ## Not available

				# Evaluate against the test set.
				print('Test Data Eval:')
				# Load the test dataset
				# test_data_dir = "data/traffic-signs/BelgiumTSC/Testing"
				test_data_dir = DEFAULT_ROOT_PATH+FLAGS['DATA_ROOT']+"/"+FLAGS['PROBLEM']+"/"+FLAGS['DATASET']+"/Testing"
				test_images, test_labels = load_data(test_data_dir)
				test_images32 = [skimage.transform.resize(image,(32,32),mode='constant') for image in test_images]
				print(test_labels)

				# display_images_and_labels(test_images32, test_labels)

				## Run predictions against the full test set.				
				predicted = do_eval(session,
					predicted_labels,
					images_ph,
					test_images32)
				# predicted = session.run([predicted_labels], feed_dict={images_ph: test_images32})[0]

				# Evaluation Metric
				# - do the evaluation on test data that is not used in training
				# - create data one for training and one for testing.
				# - then, load the test set, resize the images to 32x32, and then calculate the accuracy

				# Calculate how many matches we got
				match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
				accuracy = float(match_count) / float(len(test_labels))
				print("Accuracy: {:.3f}".format(accuracy))
				
				# Close the session. This will destroy the trained model.
				# session.close()

				# Exercise:
				# add code to save and load trained models
				# and expand to use multiple layers, convolutional networks, and data augmentation


def do_eval_viz(images,labels,predicted):
	# Visualize the Results
	# - visualization shows that the model is working or not
	# - doesn"t yet quantify how accurate it is
	# - it"s classifying the training images, so we don"t know yet if the model generalizes to images that it hasn"t seen before

	fig = plt.figure(figsize=(10, 10))
	for i in range(len(images)):
		truth = labels[i]
		prediction = predicted[i]
		print("predictions:")
		print(prediction)
		plt.subplot(5, 2,1+i)
		plt.axis('off')
		color='green' if truth == prediction else 'red'
		plt.text(40, 10, "Truth:{0}\nPrediction:{1}".format(truth, prediction),fontsize=12, color=color)
		plt.imshow(images[i])
	plt.show()


def do_eval(session,
            predicted_labels,
            images_ph,
            images):
  # Run the predicted_lables op.
  feed_dict = {images_ph: images}
  predicted = session.run([predicted_labels], feed_dict=feed_dict)[0]
  # predicted = session.run([predicted_labels], feed_dict=feed_dict)
  return predicted


def main():	
	# training_data_dir = "data/traffic-signs/BelgiumTSC/Training"
	training_data_dir = DEFAULT_ROOT_PATH+FLAGS['DATA_ROOT']+"/"+FLAGS['PROBLEM']+"/"+FLAGS['DATASET']+"/Training"
	## Load Images - training
	images, labels = load_data(training_data_dir)
	print("\nTotal: \n\tLabels: {0}\n\tImages {1}".format( len(labels), len(images) ) )
	print("\nTotal Unique: \n\tLabels: {0}".format( len(set(labels)) ) )

	# display_label_images(images,data_label)
	# display_images_and_labels(images,labels)

	#Resize images
	images32 = [skimage.transform.resize(image, (32,32), mode='constant') for image in images]
	# display_images_and_labels(images32,labels)

	# run the complete cycle on training
	run_training(images32,labels)

if __name__ == '__main__':
	main()
