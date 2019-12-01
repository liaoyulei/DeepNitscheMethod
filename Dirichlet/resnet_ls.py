'''
sys.argv[1]:
'1' means \nm{x}{2}^5
'2' means \exp(1/d\sum_{i=1}^dx_i)
'3' means \nm{x}{1}^{5/2}
beta = sys.argv[2]
'''
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape(shape[0])
        
class resnet_possion:
    def __init__(self, dimension, layer, unit, activation):
        self.dimension = dimension
        self.layer = layer
        self.unit = unit
        self.batch = 512
        self.batch_d = 16
        self.net = ResNet(layer, unit, activation)
        self.x_b = tf.compat.v1.placeholder(tf.float32, (None, dimension))
        self.x_i = tf.compat.v1.placeholder(tf.float32, (None, dimension))
        output_b = self.net(self.x_b)
        output_i = self.net(self.x_i)
        beta = float(sys.argv[2])
        self.loss_b = 2 * dimension * tf.reduce_mean((output_b - self.g(self.x_b)) ** 2)
        self.loss_i = tf.reduce_mean((self.laplace(output_i, self.x_i) + self.f(self.x_i)) ** 2)
        self.loss = beta * self.loss_b + self.loss_i
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)
        self.errl2 = self.l2(output_i - self.u(self.x_i), self.x_i) / self.l2(self.u(self.x_i), self.x_i)
        self.errh1 = self.h1(output_i - self.u(self.x_i), self.x_i) / self.h1(self.u(self.x_i), self.x_i)
        self.errh2 = self.h2(output_i - self.u(self.x_i), self.x_i) / self.h2(self.u(self.x_i), self.x_i)
        tf.compat.v1.summary.scalar('loss_b', self.loss_b)
        tf.compat.v1.summary.scalar('loss_i', self.loss_i)
        tf.compat.v1.summary.scalar('loss', self.loss)
        tf.compat.v1.summary.scalar('errl2', self.errl2)
        tf.compat.v1.summary.scalar('errh1', self.errh1)
        tf.compat.v1.summary.scalar('errh2', self.errh2)
        self.init = tf.compat.v1.global_variables_initializer()
        self.merged = tf.compat.v1.summary.merge_all()
        
    def u(self, x):
        if(sys.argv[1] == '1'):
            return tf.reduce_sum(x ** 2, axis = 1) ** 2.5
        elif(sys.argv[1] == '2'):
            return tf.exp(tf.reduce_sum(x, axis = 1) / self.dimension)
        elif(sys.argv[1] == '3'):
            return tf.reduce_sum(x, axis = 1) ** 2.5
        else:
            sys.exit("sys.argv[1] is error.")
    
    def f(self, x):
        if(sys.argv[1] == '1'):
            return -5 * (self.dimension + 3) * tf.reduce_sum(x ** 2, axis = 1) ** 1.5
        elif(sys.argv[1] == '2'):
            return -1 / self.dimension * self.u(x)
        elif(sys.argv[1] == '3'):
            return -15/4 * self.dimension * tf.reduce_sum(x, axis = 1) ** 1/2
        else:
            sys.exit("sys.argv[1] is error.")
        
    def g(self, x):
        return self.u(x)
        
    def norm2_grad(self, u, x):
        grad = tf.gradients(u, x)[0]
        return tf.reduce_sum(grad ** 2, axis = 1)

    def laplace(self, u, x):
        grad = tf.gradients(u, x)[0]
        ans = tf.zeros_like(u)
        for i in range(self.dimension):
            g = tf.gradients(grad[:, i], x)[0]
            ans += g[:, i]
        return ans    

    def l2(self, u, x):
        return tf.sqrt(tf.reduce_mean(u ** 2))
        
    def h1(self, u, x):
        return tf.sqrt(tf.reduce_mean(u ** 2 + self.norm2_grad(u, x)))
        
    def h2(self, u, x):
        ans = tf.reduce_mean(u ** 2 + self.norm2_grad(u, x))
        grad = tf.gradients(u, x)[0]
        for i in range(self.dimension):
            g = tf.gradients(grad[:, i], x)[0]
            ans += tf.reduce_mean(tf.reduce_sum(g ** 2, axis = 1))
        return tf.sqrt(ans)
        
    def train(self, sess):
        x_b = np.random.rand(2 * self.dimension * self.batch_d, self.dimension)
        for i in range(self.dimension):
            x_b[2*i*self.batch_d: (2*i+1)*self.batch_d, i] = 0.
            x_b[(2*i+1)*self.batch_d: (2*i+2)*self.batch_d, i] = 1.
        x_i = np.random.rand(self.batch, self.dimension)
        _, summary = sess.run([self.opt, self.merged], feed_dict = {self.x_b: x_b, self.x_i: x_i})
        return summary
         
    def test(self, sess, x):
        return sess.run([self.errl2, self.errh1, self.errh2], feed_dict = {self.x_i: x})

net = resnet_possion(50, 5, 100, "tanh") #d-Dimention l-Layers u-Units
with tf.compat.v1.Session() as sess:
    sess.run(net.init)
    train_writer = tf.compat.v1.summary.FileWriter('logs/', sess.graph)
    for i in range(50000):
        train_writer.add_summary(net.train(sess), i)
    errl2, errh1, errh2 = net.test(sess, np.random.rand(1000000, net.dimension))
print("d = {}, l = {}, u = {}".format(net.dimension, net.layer, net.unit))
net.net.summary()
print(sys.argv)
print("{:e} & {:e} & {:e}\\\\".format(errl2, errh1, errh2))
