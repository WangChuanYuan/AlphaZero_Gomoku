# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet with Tensorflow2

Modification from keras for compatibility:
1. Con2D's input is NHWC but not NCHW on CPU currently
2. 'EagerTensor' object has no attribute 'flatten' -> K.flatten()
3. tensor[list] -> tf.gather(tensor, list)
4. Error occurs when legal positions are unavailable
"""

from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

import numpy as np
import pickle


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height 
        self.l2_const = 1e-4  # coef of l2 penalty 
        self.create_policy_value_net()   
        self._loss_train_op()

        if model_file:
            net_params = pickle.load(open(model_file, 'rb'))
            self.model.set_weights(net_params)
        
    def create_policy_value_net(self):
        """create the policy value network """
        # TODO: Adapt for both CPU and GPU
        in_x = network = Input((4, self.board_width, self.board_height))
        network = tf.transpose(network, (0, 2, 3, 1))

        data_format = "channels_last"
        # conv layers
        network = Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format=data_format, activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format=data_format, activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format=data_format, activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        # action policy layers
        policy_net = Conv2D(filters=4, kernel_size=(1, 1), data_format=data_format, activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        policy_net = Flatten()(policy_net)
        self.policy_net = Dense(self.board_width*self.board_height, activation="softmax", kernel_regularizer=l2(self.l2_const))(policy_net)
        # state value layers
        value_net = Conv2D(filters=2, kernel_size=(1, 1), data_format=data_format, activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        value_net = Flatten()(value_net)
        value_net = Dense(64, kernel_regularizer=l2(self.l2_const))(value_net)
        self.value_net = Dense(1, activation="tanh", kernel_regularizer=l2(self.l2_const))(value_net)

        self.model = Model(in_x, [self.policy_net, self.value_net])
        
        def policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results
        self.policy_value = policy_value
        
    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()
        act_probs, value = self.policy_value(current_state.reshape(-1, 4, self.board_width, self.board_height))
        if len(legal_positions):
            act_probs = zip(legal_positions, tf.gather(K.flatten(act_probs), legal_positions))
        else:
            act_probs = zip(legal_positions, tf.convert_to_tensor([]))
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """

        # get the train op   
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=opt, loss=losses)

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def train_step(state_input, mcts_probs, winner, learning_rate):
            state_input_union = np.array(state_input)
            mcts_probs_union = np.array(mcts_probs)
            winner_union = np.array(winner)
            loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            action_probs, _ = self.model.predict_on_batch(state_input_union)
            entropy = self_entropy(action_probs)
            K.set_value(self.model.optimizer.lr, learning_rate)
            self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            return loss[0], entropy
        
        self.train_step = train_step

    def get_policy_param(self):
        net_params = self.model.get_weights()        
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
