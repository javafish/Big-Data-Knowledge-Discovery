#! /usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import datetime
import data_helpers
import word2vec_helpers
from text_cnn import TextCNN

# Parameters
# =======================================================

# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
#tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("cooking_data_file", "./data/cooking1.txt", "Data source for the cooking train data.")
tf.flags.DEFINE_string("music_data_file", "./data/music1.txt", "Data source for the music train data.")
tf.flags.DEFINE_string("video_data_file", "./data/video1.txt", "Data source for the video train data.")
tf.flags.DEFINE_integer("num_labels", 3, "Number of labels for data. (default: 3)")

#测试集
tf.flags.DEFINE_string("cooking_test","./data/cooking_test.txt","Data source for the cooking test data")
tf.flags.DEFINE_string("music_test","./data/music_test.txt","Data source for the music test data")
tf.flags.DEFINE_string("video_test","./data/video_test.txt","Data source for the video test data")


# Model hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-spearated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training paramters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evalue model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (defult: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Parse parameters from commands
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Prepare output directory for models and summaries
# =======================================================

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Data preprocess
# =======================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_positive_negative_data_files(FLAGS.cooking_data_file, FLAGS.music_data_file, FLAGS.video_data_file)

#测试集
x_test,y_test=data_helpers.load_positive_negative_data_files(FLAGS.cooking_test,FLAGS.music_test,FLAGS.video_test)
print('=============',len(x_test), len(x_test[0]))


# Get embedding vector
sentences, max_document_length = data_helpers.padding_sentences(x_text, '<PADDING>')
x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size = FLAGS.embedding_dim, file_to_save = os.path.join(out_dir, 'trained_word2vec.model')))

#测试集
sentences_test,max_document_length_test=data_helpers.padding_sentences(x_test,'<PADDING>', padding_sentence_length=18)
print(len(sentences_test), max_document_length_test)
x1=np.array(word2vec_helpers.embedding_sentences(sentences_test,embedding_size=FLAGS.embedding_dim,file_to_save=os.path.join(out_dir,'test_word2vec.model')))
print('=============x1 shape', x1.shape)

print("x.shape = {}".format(x.shape))
print("y.shape = {}".format(y.shape))

print("x_test.shape = {}".format(x1.shape))
print("y_test.shape = {}".format(y_test.shape))
# Save params
training_params_file = os.path.join(out_dir, 'training_params.pickle')
params = {'num_labels' : FLAGS.num_labels, 'max_document_length' : max_document_length}
data_helpers.saveDict(params, training_params_file)

# Shuffle data randomly
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))

#测试集
np.random.seed(10)
shuffle_indices1=np.random.permutation(np.arange(len(y_test)))


x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

#测试集
x1_shuffled=x1[shuffle_indices1]
y1_shuffled=y_test[shuffle_indices1]


# Split train/test set
# TODO: This is very crude, should use cross-validation
#dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
#x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
#y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
x_train= x_shuffled
x_dev=x1_shuffled
y_train = y_shuffled
y_dev=y1_shuffled

print("Train/Dev: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# =======================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
	log_device_placement = FLAGS.log_device_placement)
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        cnn = TextCNN(
	    sequence_length = x_train.shape[1],
	    num_classes = y_train.shape[1],
	    embedding_size = FLAGS.embedding_dim,
	    filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
	    num_filters = FLAGS.num_filters,
	    l2_reg_lambda = FLAGS.l2_reg_lambda)

	# Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, model = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.scores],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            tf_metrics(model,y_batch,sess,feed_dict)
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        def twos_like(tensor):
            size=tf.shape(tensor)
            new_tensor=tf.fill(size,2)
            return tf.to_int64(new_tensor)
        #print("hhhhhhhhhh",twos_like([1,2,3,4]))
        #求precision，recall，f1值
        def tf_metrics(model,actual_classes,session,feed_dict):
            predictions=tf.argmax(model,1)
            actuals=tf.argmax(actual_classes,1)
            print("haha",actual_classes[0],actual_classes[1])
            print("actual",actuals.eval())
            print("predictions", predictions.eval())
            twos_like_actuals=twos_like(actuals)
            ones_like_actuals = tf.ones_like(actuals)
            zeros_like_actuals = tf.zeros_like(actuals)

            twos_like_predictions = twos_like(predictions)
            ones_like_predictions = tf.ones_like(predictions)
            zeros_like_predictions = tf.zeros_like(predictions)

            tp_op_zeros = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(predictions, zeros_like_predictions)
                    ),
                    "float"
                )
            )

            tp_op_ones=tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, ones_like_actuals),
                        tf.equal(predictions, ones_like_predictions)
                    ),
                    "float"
                )
            )

            tp_op_twos = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals,tf.to_int64(twos_like_actuals)),
                        tf.equal(predictions,tf.to_int64(twos_like_predictions))
                    ),
                    "float"
                )
            )


            actual_op_zeros = tf.reduce_sum(
                tf.cast(
                    tf.equal(actuals, zeros_like_actuals)
                    ,
                    "float"
                )
            )

            actual_op_ones = tf.reduce_sum(
                tf.cast(
                    tf.equal(actuals, ones_like_actuals)
                    ,
                    "float"
                )
            )

            actual_op_twos = tf.reduce_sum(
                tf.cast(
                    tf.equal(actuals, twos_like_actuals)
                    ,
                    "float"
                )
            )

            predict_op_zeros = tf.reduce_sum(
                tf.cast(
                        tf.equal(predictions, zeros_like_predictions)
                    ,
                    "float"
                )
            )

            predict_op_ones = tf.reduce_sum(
                tf.cast(
                    tf.equal(predictions, ones_like_predictions)
                    ,
                    "float"
                )
            )

            predict_op_twos = tf.reduce_sum(
                tf.cast(
                    tf.equal(predictions, twos_like_predictions)
                    ,
                    "float"
                )
            )

            tp_zeros, predict_zeros,actual_zeros = session.run(
                [tp_op_zeros, predict_op_zeros,actual_op_zeros],
                feed_dict
            )
            tp_ones, predict_ones,actual_ones = session.run(
                [tp_op_ones, predict_op_ones,actual_op_ones],
                feed_dict
            )
            tp_twos, predict_twos,actual_twos = session.run(
                [tp_op_twos, predict_op_twos,actual_op_twos],
                feed_dict
            )

            try:
                precision_zeros=float(tp_zeros)/(float(predict_zeros))
            except ZeroDivisionError:
                precision_zeros = 0
            try:
                precision_ones = float(tp_ones) / (float(predict_ones))
            except ZeroDivisionError:
                precision_ones = 0
            try:
                precision_twos = float(tp_twos) / (float(predict_twos))
            except ZeroDivisionError:
                precision_twos = 0

            try:
                recall_zeros = float(tp_zeros) / (float(actual_zeros))
            except ZeroDivisionError:
                recall_zeros = 0
            try:
                recall_ones = float(tp_ones) / (float(actual_ones))
            except ZeroDivisionError:
                recall_ones=0
            try:
                recall_twos = float(tp_twos) / (float(actual_twos))
            except ZeroDivisionError:
                recall_twos=0


            try:
                f1_zeros = (2 * (precision_zeros * recall_zeros)) / (precision_zeros + recall_zeros)
            except ZeroDivisionError:
                f1_zeros=0
            try:
                f1_ones = (2 * (precision_ones * recall_ones)) / (precision_ones + recall_ones)
            except ZeroDivisionError:
                f1_ones=0
            try:
                f1_twos = (2 * (precision_twos * recall_twos)) / (precision_twos + recall_twos)
            except ZeroDivisionError:
                f1_twos=0

            print('Precision_zeros = ', precision_zeros)
            print('Recall_zeros = ', recall_zeros)
            print('F1 Score_zeros = ', f1_zeros)
            print('Precision_ones = ', precision_ones)
            print('Recall_ones = ', recall_ones)
            print('F1 Score_ones = ', f1_ones)
            print('Precision_twos = ', precision_twos)
            print('Recall_twos = ', recall_twos)
            print('F1 Score_twos = ', f1_twos)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
