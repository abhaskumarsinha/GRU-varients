import tensorflow as tf
import numpy as np

class CustomGRU(tf.keras.layers.Layer):
    def __init__(self, state_size = 10):
        super(CustomGRU, self).__init__()
        self.state_size = state_size
    def build(self, input_shapes):
        
        #For Update Gate: z
        self.W_update = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.U_update = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.B_update = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        
        #For Reset Gate: r
        self.W_reset = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.U_reset = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.B_reset = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        
        #For Hidden State: h_
        self.W_h_ = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.U_h_ = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.B_h_ = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        
        self.built = True
    def call(self, inputs, states):
        
        h_t_1 = states[0]
        x_t = inputs
        
        #Update Gate Equations
        update_term = tf.einsum("ij, jk -> ik", x_t, self.W_update)
        update_term = update_term + tf.einsum("ij, jk -> ik", h_t_1, self.U_update)
        ones = tf.ones((tf.shape(x_t)[0], tf.shape(x_t)[1]))
        update_term = update_term + tf.einsum("ij, jk -> ik", ones, self.B_update)
        update_term = tf.nn.sigmoid(update_term)
        
        #Reset Gate Equations
        reset_gate = tf.einsum("ij, jk -> ik", x_t, self.W_reset)
        reset_gate = reset_gate + tf.einsum("ij, jk -> ik", x_t, self.U_reset)
        ones = tf.ones((tf.shape(x_t)[0], tf.shape(x_t)[1]))
        reset_gate = reset_gate + tf.einsum("ij, jk -> ik", ones, self.B_reset)
        reset_gate = tf.nn.sigmoid(reset_gate)
        
        #h_ hidden state
        h_ = tf.einsum("ij, kj -> ik", x_t, self.W_h_)
        forget_factor = tf.einsum("ij, ij -> ij", reset_gate, h_t_1)
        h_ = h_ + tf.einsum("ij, kj -> ik", forget_factor, self.U_h_)
        ones = tf.ones((tf.shape(x_t)[0], tf.shape(x_t)[1]))
        h_ = h_ + tf.einsum("ij, jk -> ik", ones, self.B_h_)
        h_ = tf.nn.tanh(h_)
        
        
        h_t = tf.ones((tf.shape(update_term)[0], tf.shape(update_term)[1])) - update_term
        h_t = tf.einsum("ij, ij -> ij", h_t, h_t_1)
        tanh_term = tf.einsum('ij, ij -> ij', update_term, h_)
        h_t = h_t + tanh_term
        
        return h_t, [h_]

class GRU1(tf.keras.layers.Layer):
    def __init__(self, state_size = 10):
        super(GRU1, self).__init__()
        self.state_size = state_size
    def build(self, input_shapes):
        
        #For Update Gate: z
        self.W_update = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.U_update = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.B_update = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        
        #For Reset Gate: r
        self.W_reset = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.U_reset = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.B_reset = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        
        #For Hidden State: h_
        self.W_h_ = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.U_h_ = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.B_h_ = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        
        self.built = True
    def call(self, inputs, states):
        
        h_t_1 = states[0]
        x_t = inputs
        
        #Update Gate Equations
        #update_term = tf.einsum("ij, jk -> ik", x_t, self.W_update)
        update_term =  tf.einsum("ij, jk -> ik", h_t_1, self.U_update)
        ones = tf.ones((tf.shape(x_t)[0], tf.shape(x_t)[1]))
        update_term = update_term + tf.einsum("ij, jk -> ik", ones, self.B_update)
        update_term = tf.nn.sigmoid(update_term)
        
        #Reset Gate Equations
        #reset_gate = tf.einsum("ij, jk -> ik", x_t, self.W_reset)
        reset_gate = tf.einsum("ij, jk -> ik", x_t, self.U_reset)
        ones = tf.ones((tf.shape(x_t)[0], tf.shape(x_t)[1]))
        reset_gate = reset_gate + tf.einsum("ij, jk -> ik", ones, self.B_reset)
        reset_gate = tf.nn.sigmoid(reset_gate)
        
        #h_ hidden state
        h_ = tf.einsum("ij, kj -> ik", x_t, self.W_h_)
        forget_factor = tf.einsum("ij, ij -> ij", reset_gate, h_t_1)
        h_ = h_ + tf.einsum("ij, kj -> ik", forget_factor, self.U_h_)
        ones = tf.ones((tf.shape(x_t)[0], tf.shape(x_t)[1]))
        h_ = h_ + tf.einsum("ij, jk -> ik", ones, self.B_h_)
        h_ = tf.nn.tanh(h_)
        
        
        h_t = tf.ones((tf.shape(update_term)[0], tf.shape(update_term)[1])) - update_term
        h_t = tf.einsum("ij, ij -> ij", h_t, h_t_1)
        tanh_term = tf.einsum('ij, ij -> ij', update_term, h_)
        h_t = h_t + tanh_term
        
        return h_t, [h_]

class GRU2(tf.keras.layers.Layer):
    def __init__(self, state_size = 10):
        super(GRU2, self).__init__()
        self.state_size = state_size
    def build(self, input_shapes):
        
        #For Update Gate: z
        self.W_update = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.U_update = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.B_update = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        
        #For Reset Gate: r
        self.W_reset = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.U_reset = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.B_reset = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        
        #For Hidden State: h_
        self.W_h_ = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.U_h_ = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.B_h_ = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        
        self.built = True
    def call(self, inputs, states):
        
        h_t_1 = states[0]
        x_t = inputs
        
        #Update Gate Equations
        #update_term = tf.einsum("ij, jk -> ik", x_t, self.W_update)
        update_term =  tf.einsum("ij, jk -> ik", h_t_1, self.U_update)
        #ones = tf.ones((tf.shape(x_t)[0], tf.shape(x_t)[1]))
        #update_term = update_term + tf.einsum("ij, jk -> ik", ones, self.B_update)
        update_term = tf.nn.sigmoid(update_term)
        
        #Reset Gate Equations
        #reset_gate = tf.einsum("ij, jk -> ik", x_t, self.W_reset)
        reset_gate = tf.einsum("ij, jk -> ik", x_t, self.U_reset)
        #ones = tf.ones((tf.shape(x_t)[0], tf.shape(x_t)[1]))
        #reset_gate = reset_gate + tf.einsum("ij, jk -> ik", ones, self.B_reset)
        reset_gate = tf.nn.sigmoid(reset_gate)
        
        #h_ hidden state
        h_ = tf.einsum("ij, kj -> ik", x_t, self.W_h_)
        forget_factor = tf.einsum("ij, ij -> ij", reset_gate, h_t_1)
        h_ = h_ + tf.einsum("ij, kj -> ik", forget_factor, self.U_h_)
        ones = tf.ones((tf.shape(x_t)[0], tf.shape(x_t)[1]))
        h_ = h_ + tf.einsum("ij, jk -> ik", ones, self.B_h_)
        h_ = tf.nn.tanh(h_)
        
        
        h_t = tf.ones((tf.shape(update_term)[0], tf.shape(update_term)[1])) - update_term
        h_t = tf.einsum("ij, ij -> ij", h_t, h_t_1)
        tanh_term = tf.einsum('ij, ij -> ij', update_term, h_)
        h_t = h_t + tanh_term
        
        return h_t, [h_]

class GRU3(tf.keras.layers.Layer):
    def __init__(self, state_size = 10):
        super(GRU3, self).__init__()
        self.state_size = state_size
    def build(self, input_shapes):
        
        #For Update Gate: z
        self.W_update = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.U_update = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.B_update = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        
        #For Reset Gate: r
        self.W_reset = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.U_reset = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.B_reset = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        
        #For Hidden State: h_
        self.W_h_ = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.U_h_ = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        self.B_h_ = self.add_weight(shape = (input_shapes[-1], input_shapes[-1]), initializer = 'random_uniform')
        
        self.built = True
    def call(self, inputs, states):
        
        h_t_1 = states[0]
        x_t = inputs
        
        #Update Gate Equations
        #update_term = tf.einsum("ij, jk -> ik", x_t, self.W_update)
        #update_term =  tf.einsum("ij, jk -> ik", h_t_1, self.U_update)
        ones = tf.ones((tf.shape(x_t)[0], tf.shape(x_t)[1]))
        update_term = tf.einsum("ij, jk -> ik", ones, self.B_update)
        update_term = tf.nn.sigmoid(update_term)
        
        #Reset Gate Equations
        #reset_gate = tf.einsum("ij, jk -> ik", x_t, self.W_reset)
        #reset_gate = tf.einsum("ij, jk -> ik", x_t, self.U_reset)
        ones = tf.ones((tf.shape(x_t)[0], tf.shape(x_t)[1]))
        reset_gate = tf.einsum("ij, jk -> ik", ones, self.B_reset)
        reset_gate = tf.nn.sigmoid(reset_gate)
        
        #h_ hidden state
        h_ = tf.einsum("ij, kj -> ik", x_t, self.W_h_)
        forget_factor = tf.einsum("ij, ij -> ij", reset_gate, h_t_1)
        h_ = h_ + tf.einsum("ij, kj -> ik", forget_factor, self.U_h_)
        ones = tf.ones((tf.shape(x_t)[0], tf.shape(x_t)[1]))
        h_ = h_ + tf.einsum("ij, jk -> ik", ones, self.B_h_)
        h_ = tf.nn.tanh(h_)
        
        
        h_t = tf.ones((tf.shape(update_term)[0], tf.shape(update_term)[1])) - update_term
        h_t = tf.einsum("ij, ij -> ij", h_t, h_t_1)
        tanh_term = tf.einsum('ij, ij -> ij', update_term, h_)
        h_t = h_t + tanh_term
        
        return h_t, [h_]