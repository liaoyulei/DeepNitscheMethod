import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import ghalton

class ResNet(tf.keras.Model):
    def __init__(self, layer, unit, activation):
        super(ResNet, self).__init__()
        self.layer = layer
        self.fc1 = layers.Dense(unit, activation = activation)
        self.resnetlist = []
        for _ in range(layer):
            self.resnetlist.append(layers.Dense(unit, activation = activation))
            self.resnetlist.append(layers.Dense(unit, activation = activation))
        self.fc2 = layers.Dense(1, activation = None)
        
    def call(self, x):
        x = self.fc1(x)
        for i in range(self.layer):
            S = self.resnetlist[2*i](x)
            S = self.resnetlist[2*i+1](S)
            x += S
        return tf.squeeze(self.fc2(x))
        
class resnet_nitsche:
    def __init__(self, dimension, layer, unit, activation):
        self.dimension = dimension
        self.layer = layer
        self.unit = unit
        self.batch = 64
        self.batch_d = 64
        self.beta = 500
        self.sequencer = ghalton.Halton(dimension)
        self.net = ResNet(layer, unit, activation)
        self.x_b = tf.compat.v1.placeholder(tf.float64, (None, dimension))
        self.x_i = tf.compat.v1.placeholder(tf.float64, (None, dimension))
        output_b = self.net(self.x_b)
        output_i = self.net(self.x_i)
        self.loss_b = 10 * tf.reduce_mean(self.beta / 2 * (output_b - self.g(self.x_b)) ** 2 + (self.g(self.x_b) - output_b) * self.diff_n(output_b, self.x_b))
#        self.loss_b = 2 * dimension * tf.reduce_mean(self.beta / 2 * output_b ** 2 - output_b * self.diff_n(output_b, self.x_b) - self.g(self.x_b) * (self.beta * output_b - self.diff_n(output_b, self.x_b)))
        self.loss_i = 4 * tf.reduce_mean(self.norm2_grad(output_i, self.x_i) / 2 - self.f(self.x_i) * output_i)
        self.loss = self.loss_b + self.loss_i
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)
        self.errl2 = self.l2(output_i - self.u(self.x_i), self.x_i) / self.l2(self.u(self.x_i), self.x_i)
        self.errh1 = self.h1(output_i - self.u(self.x_i), self.x_i) / self.h1(self.u(self.x_i), self.x_i)
        self.errh2 = self.h2(output_i - self.u(self.x_i), self.x_i) / self.h2(self.u(self.x_i), self.x_i)
        self.init = tf.compat.v1.global_variables_initializer()
        
    def u(self, x):
        r = tf.sqrt(tf.reduce_sum(x ** 2, axis = 1))
        return tf.sqrt((r - x[:, 0]) / 2)
    
    def f(self, x):
        return 0
        
    def g(self, x):
        return self.u(x)
        
    def norm2_grad(self, u, x):
        grad = tf.gradients(u, x)[0]
        return tf.reduce_sum(grad ** 2, axis = 1)

    def diff_n(self, u, x):
        grad = tf.gradients(u, x)[0]
        difflist = []
        for i in range(self.dimension):
            difflist.append(-grad[4*i*self.batch_d: (4*i+2)*self.batch_d, i])
            difflist.append(grad[(4*i+2)*self.batch_d: (4*i+4)*self.batch_d, i])
        difflist.append(-grad[8*self.batch_d: 9*self.batch_d, 1]);
        difflist.append(grad[9*self.batch_d:, 1])
        return tf.concat(difflist, axis = 0)

    def laplace(self, u, x):
        grad = tf.gradients(u, x)[0]
        ans = tf.zeros_like(u)
        for i in range(self.dimension):
            g = tf.gradients(grad[:, i], x)[0]
            ans += g[:, i]
        return ans    

    def l2(self, u, x):
        return tf.sqrt(4 * tf.reduce_mean(u ** 2))
        
    def h1(self, u, x):
        return tf.sqrt(4 * tf.reduce_mean(u ** 2 + self.norm2_grad(u, x)))
        
    def h2(self, u, x):
        ans = 4 * tf.reduce_mean(u ** 2 + self.norm2_grad(u, x))
        grad = tf.gradients(u, x)[0]
        for i in range(self.dimension):
            g = tf.gradients(grad[:, i], x)[0]
            ans += 4 * tf.reduce_mean(tf.reduce_sum(g ** 2, axis = 1))
        return tf.sqrt(ans)
        
    def train(self, sess):
        x_b = np.array(self.sequencer.get(10 * self.batch_d))
        for i in range(self.dimension):
            x_b[4*i*self.batch_d: (4*i+1)*self.batch_d, i] = -1.
            x_b[4*i*self.batch_d: (4*i+1)*self.batch_d, 1-i] -= 1.
            x_b[(4*i+1)*self.batch_d: (4*i+2)*self.batch_d, i] = -1.
            x_b[(4*i+2)*self.batch_d: (4*i+3)*self.batch_d, i] = 1.
            x_b[(4*i+2)*self.batch_d: (4*i+3)*self.batch_d, 1-i] -= 1.
            x_b[(4*i+3)*self.batch_d: (4*i+4)*self.batch_d, i] = 1.
        x_b[8*self.batch_d: , 1] = 0.
        x_i = np.array(self.sequencer.get(4 * self.batch))
        x_i[: 2*self.batch, 0] -= 1.
        x_i[self.batch: 3*self.batch, 1] -= 1.
        sess.run(self.opt, feed_dict = {self.x_b: x_b, self.x_i: x_i})
         
    def test(self, sess, batch_t):
        x_i = np.array(self.sequencer.get(4 * batch_t))
        x_i[: 2*batch_t, 0] -= 1.
        x_i[batch_t: 3*batch_t, 1] -= 1.
        return sess.run([self.errl2, self.errh1, self.errh2], feed_dict = {self.x_i: x_i})

net = resnet_nitsche(2, 5, 10, "tanh") #dimension, layer, unit
L2, H1 = [], []
with tf.compat.v1.Session() as sess:
    sess.run(net.init)
    for i in range(50000):
        net.train(sess)
        if i % 100 == 0:
            errl2, errh1, errh2 = net.test(sess, 100000)
            L2.append(errl2)
            H1.append(errh1)
print("d = {}, l = {}, u = {}".format(net.dimension, net.layer, net.unit))
net.net.summary()
print(L2)
print(H1)

