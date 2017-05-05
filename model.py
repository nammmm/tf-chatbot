""" A neural chatbot using sequence to sequence model with
attentional decoder. 

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

This file contains the code to build the model
"""
from __future__ import print_function

import random

import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import config

class Seq2SeqModel(object):
    def __init__(self, forward_only, batch_size, dtype=tf.float32):
        """forward_only: if set, we do not construct the backward pass in the model.
        """
        print('Initialize new model')
        self.fw_only = forward_only
        self.batch_size = batch_size
        self.dtype=dtype
    
    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create placeholders')
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in xrange(config.BUCKETS[-1][0])] # Last bucket is the biggest one.
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in xrange(config.BUCKETS[-1][1] + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in xrange(config.BUCKETS[-1][1] + 1)]

        # Our targets are decoder inputs shifted by one (to ignore <s> symbol)
        self.targets = self.decoder_inputs[1:]
        
    def _inference(self):
        print('Create inference')
        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            w_t = tf.get_variable("proj_w", [config.DEC_VOCAB, config.HIDDEN_SIZE], dtype=self.dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable('proj_b', [config.DEC_VOCAB], dtype=self.dtype)
            self.output_projection = (w, b)

        def sampled_loss(labels, logits):
            labels = tf.reshape(labels, [-1, 1])
            # We need to compute the sampled_softmax_loss using 32bit floats to
            # avoid numerical instabilities.
            local_w_t = tf.cast(w_t, tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(logits, tf.float32)
            return tf.cast(
            tf.nn.sampled_softmax_loss(
                weights=local_w_t,
                biases=local_b,
                labels=labels,
                inputs=local_inputs,
                num_sampled=config.NUM_SAMPLES,
                num_classes=config.DEC_VOCAB),
            self.dtype)
        self.softmax_loss_function = sampled_loss
        
        def single_cell():
            return tf.contrib.rnn.BasicLSTMCell(config.HIDDEN_SIZE)
        self.cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(config.NUM_LAYERS)])

    def _create_loss(self):
        print('Creating loss... \nIt might take a couple of minutes depending on how many buckets you have.')
        start = time.time()
        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    self.cell,
                    num_encoder_symbols=config.ENC_VOCAB,
                    num_decoder_symbols=config.DEC_VOCAB,
                    embedding_size=config.HIDDEN_SIZE,
                    output_projection=self.output_projection,
                    feed_previous=do_decode,
                    dtype=self.dtype)

        if self.fw_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, self.targets,
                self.decoder_masks, config.BUCKETS, lambda x, y: _seq2seq_f(x, y, True),
                softmax_loss_function=self.softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection:
                for bucket in xrange(len(config.BUCKETS)):
                    self.outputs[bucket] = [
                        tf.matmul(output, self.output_projection[0]) + self.output_projection[1]
                        for output in self.outputs[bucket]
                    ]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, self.targets,
                self.decoder_masks, config.BUCKETS,
                lambda x, y: _seq2seq_f(x, y, False),
                softmax_loss_function=self.softmax_loss_function)
        print('Time:', time.time() - start)

    def _creat_optimizer(self):
        print('Create optimizer... \nIt might take a couple of minutes depending on how many buckets you have.')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            # Gradients and SGD update operation for training the model.
            params = tf.trainable_variables()
            if not self.fw_only:
                self.gradient_norms = []
                self.train_ops = []
                self.optimizer = tf.train.GradientDescentOptimizer(config.LR)
                start = time.time()
                for bucket in xrange(len(config.BUCKETS)):
                    gradients = tf.gradients(self.losses[bucket], params)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                    config.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(
                        zip(clipped_gradients, params), global_step=self.global_step))
                    print('Creating opt for bucket {} took {} seconds'.format(bucket, time.time() - start))
                    start = time.time()

    def _create_summary(self):
        pass

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._creat_optimizer()
        self._create_summary()