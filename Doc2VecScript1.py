import numpy as np
import spacy, os, re, io
import gensim.models.word2vec as w2v
from gensim.corpora import Dictionary
import multiprocessing
import math
import tensorflow as tf
import time
from datetime import timedelta
from collections import namedtuple

############################# Reading Questions #############################

question_dir = 'LabelledData.txt'

questions = []
types = []
with io.open(question_dir, errors='ignore', encoding='utf-8') as fid:
	for line in fid:
		question, type = line.split(",,,")
		question = question.strip()
		type = type.strip()
		questions.append(question)
		types.append(type)

############################# Preprocessing Questions #############################

#create word_list
nlp = spacy.load('en')
processed_questions = []
for question in nlp.pipe(questions, n_threads=4, batch_size=10):
	question = [token.lemma_ for token in question if token.is_alpha and len(token)>1]
	processed_questions.append(question)

questions = processed_questions
del processed_questions

# Compute bigrams.
from gensim.models import Phrases

# Add bigrams to question (only ones that appear 3 times or more).
bigram = Phrases(questions, min_count=3)

for idx in range(len(questions)):
	for token in bigram[questions[idx]]:
		if '_' in token:
			# Token is a bigram, add to question.
			questions[idx].append(token)
			

processed_questions = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')

for i, text in enumerate(questions):
	tags = [i]
	processed_questions.append(analyzedDocument(text, tags))

questions = processed_questions
del processed_questions


######################################### Create Doc2vec #########################################

from gensim.models import doc2vec

#Hyper Parameters for doc2vec Model
num_workers = multiprocessing.cpu_count()
context_size = 5
num_features = 50
min_word_Count = 1

model = doc2vec.Doc2Vec(questions, size = num_features, window = context_size, min_count = min_word_Count, workers = num_workers)

############################# create ques Vector ###################################

#questions_vec = np.array((len(questions), 50))
questions_vec = []

#storing all vectors of questions
for i in range(len(questions)):
	vec = model.docvecs[i]
	questions_vec.append(vec)

############################# create labels ###################################


question_type = ["who", "what", "when", "affirmation", "unknown"]

n_samples = len(questions)
n_labels = len(question_type)

questions_label = np.zeros((n_samples,  n_labels))

for i, type in enumerate(types):
	questions_label[i][question_type.index(type)] = 1


############################# create training and test data ###################################

def get_train_test_inds(y,train_proportion=0.8):
	train_inds = np.zeros(len(y),dtype=bool)
    	test_inds = np.zeros(len(y),dtype=bool)
	values = list(set(y))
	for value in values:
        	value_inds = []
        	for i, x in enumerate(y):
			if x == value:
				value_inds.append(i)
		np.random.shuffle(value_inds)
        	n = int(train_proportion*len(value_inds))
		train_inds[value_inds[:n]]=True
        	test_inds[value_inds[n:]]=True

	return train_inds,test_inds

train_inds,test_inds = get_train_test_inds(types)

train_ques = []
train_label = []

for i, idx in enumerate(train_inds):
	if idx==True:
		train_ques.append(questions_vec[i])
		train_label.append(questions_label[i])

test_ques = []
test_label = []

for i, idx in enumerate(test_inds):
	if idx==True:
		test_ques.append(questions_vec[i])
		test_label.append(questions_label[i])

test_true = tf.placeholder(tf.float32, shape=[None, n_labels], name='test_true')

test_true_cls = tf.argmax(test_true, axis=1)


########################## Architecture of 2 hidden layer NN ###################################

#number of node in layer 1
num_nodeOne = 20

#number of node in layer 2
num_nodeTwo = 20

learning_rate=1e-4

x = tf.placeholder(tf.float32, [None, num_features], name = 'x')

y_true = tf.placeholder(tf.float32, shape=[None, n_labels], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)


W1 = tf.Variable(tf.truncated_normal(shape=[num_features, num_nodeOne], stddev=0.05))
b1 = tf.Variable(tf.constant(0.05, shape=[num_nodeOne]))

layer1 = tf.add(tf.matmul(x, W1), b1)

W2 = tf.Variable(tf.truncated_normal(shape=[num_nodeOne, num_nodeTwo], stddev=0.05))
b2 = tf.Variable(tf.constant(0.05, shape=[num_nodeTwo]))

layer2 = tf.add(tf.matmul(layer1, W2), b2)

W3 = tf.Variable(tf.truncated_normal(shape=[num_nodeTwo, n_labels], stddev=0.05))
b3 = tf.Variable(tf.constant(0.05, shape=[n_labels]))

y_values = tf.add(tf.matmul(layer2, W3), b3)

y_pred = tf.nn.softmax(y_values)

y_pred_cls = tf.argmax(y_pred, axis=1)

#cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_values, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

#Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Performance Measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

#accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

########################## Initialise sesssion ###################################

# Initialize variabls and tensorflow session
session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64

########################  Helper-function to perform optimization iterations ####################### 

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
	# Ensure we update the global variable rather than a local copy.
   	global total_iterations

    	# Start-time used for printing time-usage below.
    	start_time = time.time()

   	for i in range(total_iterations, total_iterations + num_iterations):
        	# Put the batch into a dict with the proper names
        	# for placeholder variables in the TensorFlow graph.
        	feed_dict_train = {x: train_ques, y_true: train_label}

        	# Run the optimizer using this batch of training data.
        	# TensorFlow assigns the variables in feed_dict_train
        	# to the placeholder variables and then runs the optimizer.
        	session.run(optimizer, feed_dict=feed_dict_train)

        	# Print status every 100 iterations.
        	if i % 100 == 0:
            		# Calculate the accuracy on the training-set.
            		acc = session.run(accuracy, feed_dict=feed_dict_train)

            		# Message for printing.
            		msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            		# Print it.
            		print(msg.format(i + 1, acc))

    	# Update the total number of iterations performed.
    	total_iterations += num_iterations

    	# Ending time.
    	end_time = time.time()

    	# Difference between start and end-times.
    	time_dif = end_time - start_time

    	# Print the time-usage.
    	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


######################## Helper-function for showing the classification accuracy #######################

# Split the test-set into smaller batches of this size.
test_batch_size = 10

def print_test_accuracy():
		# Number of question in the test-set.
    	num_test = len(test_ques)

    	# Allocate an array for the predicted classes which
    	# will be calculated in batches and filled into this array.
    	cls_pred = np.zeros(shape=num_test, dtype=np.int)

    	# Now calculate the predicted classes for the batches.
    	# We will just iterate through all the batches.
    	# There might be a more clever and Pythonic way of doing this.
	
	
    	# The starting index for the next batch is denoted i.
	feed_dict = {x: test_ques, y_true: test_label}
	cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)

    	# Convenience variable for the true class-numbers of the test-set.
	feed_dict = {test_true: test_label}
	cls_true = session.run(test_true_cls, feed_dict=feed_dict)

    	# Create a boolean array whether each question is correctly classified.
    	correct = (cls_true == cls_pred)

    	# Calculate the number of correctly classified questions.
    	# When summing a boolean array, False means 0 and True means 1.
    	correct_sum = correct.sum()

    	# Classification accuracy is the number of correctly classified
    	# questions divided by the total number of questions in the test-set.
    	acc = float(correct_sum) / num_test

    	# Print the accuracy.
    	msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    	print(msg.format(acc, correct_sum, num_test))

######################## Running the model and Checking Accuracy #######################

#Performance before any optimization
print_test_accuracy()


#Performance after 1 optimization iteration
optimize(num_iterations=1)
print_test_accuracy()


#Performance after 100 optimization iterations
optimize(num_iterations=99) # We already performed 1 iteration above.
print_test_accuracy()


#Performance after 1000 optimization iterations
optimize(num_iterations=900)  #We performed 100 iterations above.
print_test_accuracy() 

#Performance after 10000 optimization iterations
#optimize(num_iterations=9000)  #We performed 1000 iterations above.
#print_test_accuracy()
######################## Taking user inputs and checking model #######################


choice = 'yes'
while choice=='yes' or choice=='y' :
	print "Enter a Question"
	input1 = raw_input().decode('utf8')
	question = nlp(input1)
	question = [token.lemma_ for token in question if token.is_alpha and len(token)>1]
	ques_vec = [model.infer_vector(question)]
	feed_dict = {x: ques_vec}
	cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)
	print "Question Type: ", question_type[cls_pred[0]]
	print "Want to check again (y/n)"
	choice = raw_input()
	