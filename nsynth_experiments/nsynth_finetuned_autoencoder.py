from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from tensorflow.python import pywrap_tensorflow
slim = tf.contrib.slim

from model_units import *
import librispeech_reader 

class Config(object):
    def __init__(self, train_path = None):
        self.num_iters = 200000
        self.learninig_rate = 1e-3
        self.ae_hop_length = 512
        self.ae_bottleneck_width = 16
        self.train_path = train_path
        
    def get_batch(self, batch_size):
        print('Get batch')
        assert self.train_path is not None
        data_train = librispeech_reader.LibriSpeechDataset(
            self.train_path, is_training = True)
        return data_train.get_wavenet_batch(batch_size, length = 39936)
    
    @staticmethod
    def _condition(x, encoding):
        """Condition the input on the encoding.
        Args:
          x: The [mb, length, channels] float tensor input.
          encoding: The [mb, encoding_length, channels] float tensor encoding.
        Returns:
          The output after broadcasting the encoding to x's shape and adding them.
        """
        mb, length, channels = x.get_shape().as_list()
        enc_mb, enc_length, enc_channels = encoding.get_shape().as_list()
        assert enc_mb == mb
        assert enc_channels == channels

        encoding = tf.reshape(encoding, [mb, enc_length, 1, channels])
        x = tf.reshape(x, [mb, enc_length, -1, channels])
        x += encoding
        x = tf.reshape(x, [mb, length, channels])
        x.set_shape([mb, length, channels])
        return x
    
    
    
    
    def build(self, inputs, is_training):
        ''' 
        Args:
          inputs: A dict of inputs. For training, should contain 'wav'.
        '''
        del is_training
        num_stages = 10 
        num_layers = 20
        filter_length = 3
        width = 512
        ae_num_stages = 10
        ae_num_layers = 20
        ae_filter_length = 3
        ae_width = 128
        skip_width = 256
        
        # Encode the source with 8-bit Mu-Law.
        x = inputs['wav']
        x_quantized = mu_law(x)
        with tf.name_scope("scale_and_expand_dims"):
            x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
            x_scaled = tf.expand_dims(x_scaled, 2)
            
        
        ###
        # The Non-Causal Temporal Encoder.
        ###
        en = conv1d(x_scaled,
                    causal=False,
                    num_filters=ae_width,
                    filter_length=ae_filter_length, dilation = 1,
                    name='ae_startconv')

        for num_layer in range(ae_num_layers):
            with tf.name_scope('layer_{}'.format(num_layer)):
                dilation = 2**(num_layer % ae_num_stages)

                d = tf.nn.relu(en)
                d = conv1d(d,
                           causal=False,
                           num_filters=ae_width,
                           filter_length=ae_filter_length,
                           dilation=dilation,
                           name='ae_dilatedconv_%d' % (num_layer + 1))
                d = tf.nn.relu(d)
                en += conv1d(d,
                             num_filters=ae_width,
                             filter_length=1,
                             name='ae_res_%d' % (num_layer + 1))

        en = conv1d(en,
                    num_filters=self.ae_bottleneck_width,
                    filter_length=1,
                    name='ae_bottleneck')
        en = pool1d(en, self.ae_hop_length, name = 'ae_pool', mode = 'avg')
        encoding = en
                    
        
        ###
        # Classification layers.
        ###
        en_shape = encoding.get_shape().as_list()
        clf_hu_size1 = 256
        clf_hu_size2 = 128
        clf_output_size = 40
        
        clf = tf.reshape(en, [-1, en_shape[-1]*en_shape[-2]])
        clf = fc_layer(clf,  en_shape[-1]*en_shape[-2], clf_hu_size1, "clf_fc1")
        clf = tf.nn.relu(clf, name = "clf_relu1")
        clf = tf.nn.dropout(clf, keep_prob = 0.5)
        clf = fc_layer(clf,  clf_hu_size1, clf_hu_size2, "clf_fc2")
        clf = tf.nn.relu(clf, name = "clf_relu2")
        
        
             ###
        # The WaveNet Decoder.
        ###
        l = shift_right(x_scaled)
        l = conv1d(l, num_filters=width, filter_length=filter_length, name='startconv')

        # Set up skip connections.
        s = conv1d(l, num_filters=skip_width, filter_length=1, name='skip_start')

        # Residual blocks with skip connections.
        for i in range(num_layers):
            dilation = 2**(i % num_stages)
            d = conv1d(
              l,
              num_filters=2 * width,
              filter_length=filter_length,
              dilation=dilation,
              name='dilatedconv_%d' % (i + 1))
            d = self._condition(d,conv1d(
                                  en,
                                  num_filters=2 * width,
                                  filter_length=1,
                                  name='cond_map_%d' % (i + 1)))

            assert d.get_shape().as_list()[2] % 2 == 0
            m = d.get_shape().as_list()[2] // 2
            d_sigmoid = tf.sigmoid(d[:, :, :m])
            d_tanh = tf.tanh(d[:, :, m:])
            d = d_sigmoid * d_tanh

            l += conv1d(
              d, num_filters=width, filter_length=1, name='res_%d' % (i + 1))
            s += conv1d(
              d, num_filters=skip_width, filter_length=1, name='skip_%d' % (i + 1))

        s = tf.nn.relu(s)
        s = conv1d(s, num_filters=skip_width, filter_length=1, name='out1')
        s = self._condition(s, conv1d(
                                en,
                                num_filters=skip_width,
                                filter_length=1,
                                name='cond_map_out1'))
        s = tf.nn.relu(s)

        ###
        # Compute the logits and get the loss for classifier.
        ###
        logits1 = fc_layer(clf,  clf_hu_size2, clf_output_size, "clf_logits1")
        probs1 = tf.nn.softmax(logits1, name = 'clf_softmax1')

#         logits2 = fc_layer(clf,  clf_hu_size2, 2, "clf_fc2")
#         probs2 = tf.nn.softmax(logits2, name = 'clf_softmax2')

     
        with tf.name_scope('clf_loss'):
            loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits1, labels=inputs['label'], name='nll'), 0, name='clf_loss1')

#             loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
#                 logits=logits2, labels=inputs['sex'], name='nll'), 0, name='loss2')

#             loss = loss1
#             loss = loss1 + loss2

        
        ###
        # Compute the logits and get the loss for decoder. 
        ###
        logits_dec = conv1d(s, num_filters=256, filter_length=1, name='dec_logits')
        logits_dec = tf.reshape(logits_dec, [-1, 256])
        probs_dec = tf.nn.softmax(logits_dec, name='dec_softmax')
        x_indices = tf.cast(tf.reshape(x_quantized, [-1]), tf.int32) + 128
        loss_dec = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_dec, labels=x_indices, name='nll'),
            0,
            name='dec_loss')
        
        loss = loss1 + loss_dec

        return {'predictions_speaker' : logits1,
#                 'predictions_sex' : logits2,
                'loss': loss,
                'eval': {'nll':loss}, 
                'quantized_input' : x_quantized,
                'encoding':encoding,
               }
    
    
if __name__=="__main__":
    LOGDIR = "/workspace/models_pretrained/nsynth_finetuned_autoencoder/"
    OLD_MODEL_PATH = '/workspace/models_pretrained/wavenet-ckpt/model.ckpt-200000'
    
    # "Batch size spread across all sync replicas.We use a size of 32."
    TOTAL_BATCH_SIZE = 1

    # "Number of replicas. We train with 32."
    WORKER_REPLICAS = 1

    # "Number of tasks in the ps job. If 0 no ps job is used. We typically use 11 "
    PS_TASKS = 0

    wav_path = '/workspace/data/LibriSpeech_to_classify/train'
    tfrecord_path = '{}/wavs.tfrecord'.format(wav_path)

    # "The path to the train tfrecord"
    TRAIN_PATH = tfrecord_path 

    #  "Task id of the replica running the training."
    TASK = 0
    
    
    model = Config(train_path = TRAIN_PATH)   

    with tf.Graph().as_default():
        logdir = LOGDIR
        total_batch_size = TOTAL_BATCH_SIZE
        worker_replicas = WORKER_REPLICAS
        assert total_batch_size % worker_replicas == 0
        worker_batch_size = int(total_batch_size / worker_replicas)


        # Run the Reader on the CPU
        cpu_device = "/job:localhost/replica:0/task:0/CPU:0"
        if PS_TASKS:
            cpu_device = "/job:worker/CPU:0"

        # Get input batch for worker
        with tf.device(cpu_device):
            inputs_dict = model.get_batch(worker_batch_size)


        with tf.device( tf.train.replica_device_setter(
            ps_tasks=PS_TASKS, merge_devices=True)):

            global_step = tf.get_variable(
                "global_step", [],
                tf.int32,
                initializer=tf.constant_initializer(0),
                trainable=False)

            lr = model.learninig_rate
            tf.summary.scalar("learning_rate", lr)

            print('Lets get output')
            outputs_dict = model.build(inputs_dict, is_training = True)
            loss = outputs_dict["loss"]
            tf.summary.scalar("train_loss", loss)

    #         # if we don't want to use moving average while averaging gradients from replicas 
    #         opt = tf.train.SyncReplicasOptimizer(
    #             tf.train.AdamOptimizer(lr, epsilon = 1e-8),
    #             replicas_to_aggregate = worker_replicas,
    #             total_num_replicas = worker_replicas)

            # if we want to use moving average while averaging gradients from replicas 
            ema = tf.train.ExponentialMovingAverage(
              decay=0.9999, num_updates=global_step)
            opt = tf.train.SyncReplicasOptimizer(
                tf.train.AdamOptimizer(lr, epsilon=1e-8),
                replicas_to_aggregate = worker_replicas,
                total_num_replicas=worker_replicas,
                variable_averages=ema,
                variables_to_average=tf.trainable_variables())

            train_op = opt.minimize(
                loss,
                global_step = global_step,
                name = "train",
                colocate_gradients_with_ops = True)

            session_config = tf.ConfigProto(allow_soft_placement = True)

            ###
            # Initialize ae layers with corresponding weights from pre-trained wavenet autoencoder
            trainable_var_names = [v.op.name for v in tf.trainable_variables()]

            var_names_to_values = {}
            reader = pywrap_tensorflow.NewCheckpointReader(OLD_MODEL_PATH) 

            for key in np.intersect1d(trainable_var_names, list(reader.get_variable_to_shape_map().keys())):
                var_names_to_values[key] = reader.get_tensor(key)

            init_assign_op, init_feed_dict = slim.assign_from_values(var_names_to_values)

            def InitAssignFn(sess):
                sess.run(init_assign_op, init_feed_dict)
            ###

            is_chief = (TASK == 0)
            local_init_op = opt.chief_init_op if is_chief else opt.local_step_init_op

            slim.learning.train(
                train_op=train_op,
                is_chief=is_chief,
                logdir = logdir,
                number_of_steps=model.num_iters,
                global_step=global_step,
                log_every_n_steps=250,
                local_init_op=local_init_op,
                save_interval_secs=300,
                sync_optimizer=opt,
                session_config=session_config,
                init_fn = InitAssignFn)
