import numpy as np
import os
import tensorflow as tf
from time import time
from tqdm import tqdm
from .layers import (_causal_linear, _output_linear, conv1d,
                    dilated_conv1d)


class Model(object):
    def __init__(self,
                 #wav,
                 num_time_samples,
                 num_channels=1,
                 num_classes=256, # Quantification, doesn't impact complexity while up to 256
                 num_blocks=2, # Number of stack
                 num_layers=16, # To test from 10 to 16
                 num_hidden=256, # To test from 64 to 512
                 gpu_fraction=1.0):

        #self.wav = wav
        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gpu_fraction = gpu_fraction

        tf.compat.v1.reset_default_graph()

        inputs = tf.compat.v1.placeholder(tf.float32,
                                shape=(None, num_time_samples, num_channels), name = 'inputs')
        targets = tf.compat.v1.placeholder(tf.int32, shape=(None, num_time_samples), name = 'targets')


        h = inputs
        hs = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                h = dilated_conv1d(h, num_hidden, rate=rate, name=name)
                hs.append(h)

        outputs = conv1d(h,
                         num_classes,
                         filter_width=1,
                         gain=1.0,
                         activation=None,
                         bias=True)

        costs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=outputs, labels=targets)
        cost = tf.reduce_mean(costs)

        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)

        saver = tf.compat.v1.train.Saver(max_to_keep = 1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.compat.v1.global_variables_initializer())

        self.inputs = inputs
        self.targets = targets
        self.outputs = outputs
        self.hs = hs
        self.costs = costs
        self.cost = cost
        self.train_step = train_step
        self.sess = sess
        self.saver = saver

    def _train(self, inputs, targets):
        feed_dict = {self.inputs: inputs, self.targets: targets}
        #with tf.Session as sess :
        cost, _ = self.sess.run(
            [self.cost, self.train_step],
            feed_dict=feed_dict)
        return cost

    def train(self, inputs, targets, stopcriterion = 0.1):
            losses = []
            terminal = False
            i = 0
            max_step=5000
            bar = tqdm(total = max_step,  initial = i)
            tic=time()
            ckpt_path = 'B{}_L{}_D{}'.format(self.num_blocks, self.num_layers, self.num_hidden)
            print('Training started')
            while not terminal:
                i += 1
                bar.update(1)
                cost = self._train(inputs, targets)
                try:
                    if cost < stopcriterion or i == max_step:
                        print('The final cost is ' + str(cost))
                        if os.path.isdir('checkpoints/' + ckpt_path ) == False :
                            os.mkdir('checkpoints/' + ckpt_path )
                        saved_path = self.saver.save(self.sess, 'checkpoints/' + ckpt_path + '/' + ckpt_path)
                        print('model saved in : {}'.format(saved_path))
                        terminal = True
                    losses.append(cost)
                    if i % 10 == 0 :
                        print('The cost at step {} is {}'.format(i, cost))
                    if i % 200 == 0 :
                        if os.path.isdir('checkpoints/' + ckpt_path ) == False :
                            os.mkdir('checkpoints/' + ckpt_path )
                        saved_path = self.saver.save(self.sess, 'checkpoints/' + ckpt_path + '/' + ckpt_path)
                        print('Step {}, model saved in : {}'.format(i, saved_path))
                except KeyboardInterrupt:
                    print('The final cost is ' + str(cost))
                    if os.path.isdir('checkpoints/' + ckpt_path ) == False :
                        os.mkdir('checkpoints/' + ckpt_path )
                    saved_path = self.saver.save(self.sess, 'checkpoints/' + ckpt_path + '/' + ckpt_path)
                    print('Interrupted... Model saved in : {}'.format(saved_path))
                    terminal = True
                    pass
                #if i % 100 == 0 :
                #    if os.path.isdir('checkpoints/' + ckpt_path ) == False :
                #        os.mkdir('checkpoints/' + ckpt_path )
                #    saved_path = self.saver.save(self.sess, 'checkpoints/' + ckpt_path  + '/' + ckpt_path, global_step = i)
                #    print('model saved in : {}'.format(saved_path))
            bar.close()
            return(i, losses)

    def restore(self):
        path = 'B{}_L{}_D{}'.format(self.num_blocks, self.num_layers, self.num_hidden)
        try:
            print('Importing meta-graph...')
            self.saver = tf.compat.v1.train.import_meta_graph('checkpoints/' + path + '/' + path + '.meta')
            self.sess = tf.compat.v1.Session()
            print('Restoring checkpoint...')
            self.saver.restore(self.sess, 'checkpoints/' + path + '/' + path)
            print('Model {} restored !'.format(path))
        except Exception as e:
            print(str(e))
            print('Unable to restore the model')


class Generator(object):
    def __init__(self, model, input_size = 1):

        self.model = model
        batch_size = 1
        self.bins = np.linspace(-1, 1, self.model.num_classes)

        inputs = tf.compat.v1.placeholder(tf.float32, [batch_size, input_size], name='inputs')

        count = 0
        h = inputs

        init_ops = []
        push_ops = []
        for b in range(self.model.num_blocks):
            for i in range(self.model.num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                if count == 0:
                    state_size = 1
                else:
                    state_size = self.model.num_hidden

                q = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, state_size))
                init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))

                state_ = q.dequeue()
                push = q.enqueue([h])
                init_ops.append(init)
                push_ops.append(push)

                h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)

                count += 1

        outputs = _output_linear(h)

        out_ops = [tf.nn.softmax(outputs)]
        out_ops.extend(push_ops)

        self.inputs = inputs
        self.init_ops = init_ops
        self.out_ops = out_ops

        # Initialize queues.
        self.model.sess.run(self.init_ops)

    def run(self, input, num_samples):
        predictions = []
        for step in tqdm(range(num_samples)):

            feed_dict = {self.inputs: input}
            output = self.model.sess.run(self.out_ops, feed_dict=feed_dict)[0] # Ignore push_ops
            value = np.argmax(output[0, :])

            input = np.array(self.bins[value])[None, None]
            predictions.append(input)

        predictions_ = np.concatenate(predictions, axis = 1)
        return predictions_
