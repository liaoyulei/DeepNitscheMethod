'''
sys.argv[1]:
'1' means \sin(\pi x)\sin(\pi y)
'2' means x^3y^2+x\sin(2\pi xy)\sin(2\pi y)
'3' means \exp((x^2+y^2)/2)+\sin(2\pi(x+y))\sin(2\pi y)
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
            S = self.resnetlist[2*i+1](x)
            x += S
        return tf.squeeze(self.fc2(x))
        
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape(shape[0])
        
class resnet_ritz:
    def __init__(self, dimension, layer, unit, activation):
        self.dimension = dimension
        self.layer = layer
        self.unit = unit
        self.batch = 128
        self.batch_d = self.batch // (2 * dimension) + 1
        self.net = ResNet(layer, unit, activation)
        self.x_d = tf.compat.v1.placeholder(tf.float32, (None, dimension))
        self.x_n = tf.compat.v1.placeholder(tf.float32, (None, dimension))
        self.x_i = tf.compat.v1.placeholder(tf.float32, (None, dimension))
        output_d = self.net(self.x_d)
        output_n = self.net(self.x_n)
        output_i = self.net(self.x_i)
        beta = float(sys.argv[2])
        self.loss_b = self.dimension * (beta * tf.reduce_mean((output_d - self.g_d(self.x_d)) ** 2) - tf.reduce_mean(self.g_n(self.x_n) * output_n))
        self.loss_i = tf.reduce_mean(self.agradgrad(output_i, self.x_i) / 2 - self.f(self.x_i) * output_i)
        self.loss = self.loss_b + self.loss_i
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
            return tf.reduce_prod(tf.sin(np.pi * x), axis = 1)
        elif(sys.argv[1] == '2'):
            return x[:, 0] * tf.reduce_prod(x ** 2, axis = 1) + x[:, 0] * tf.sin(2 * np.pi * tf.reduce_prod(x, axis = 1)) * tf.sin(2 * np.pi * x[:, 1])
        elif(sys.argv[1] == '3'):
            return tf.exp(tf.reduce_sum(x ** 2, axis = 1) / 2) + tf.sin(2 * np.pi * tf.reduce_sum(x, axis = 1)) * tf.sin(2 * np.pi * x[:, 1])
        else:
            sys.exit("sys.argv[1] is error.")
    
    def a11(self, x):
        if(sys.argv[1] == '1'):
            return 2 + tf.sin(np.pi * x[:, 0]) * tf.cos(np.pi * x[:, 1])
        elif(sys.argv[1] == '2'):
            return (x[:, 0] + 1) ** 2 + x[:, 1] ** 2
        elif(sys.argv[1]) == '3':
            return 3 + tf.cos(2 * np.pi * x[:, 0])
     
    def a12(self, x):
        if(sys.argv[1] == '1'):
            return 0
        elif(sys.argv[1] == '2'):
            return -tf.reduce_prod(x, axis = 1)
        elif(sys.argv[1] == '3'):
            return x[:, 0] - x[:, 1]
        else:
            sys.exit("sys.argv[1] is error.")
    
    def a21(self, x):
        if(sys.argv[1] == '1'):
            return 0
        elif(sys.argv[1] == '2'):
            return -tf.reduce_prod(x, axis = 1)
        elif(sys.argv[1] == '3'):
            return x[:, 0] - x[:, 1]
        else:
            sys.exit("sys.argv[1] is error.")
            
    def a22(self, x):
        if(sys.argv[1] == '1'):
            return 2 + tf.sin(np.pi * x[:, 0]) * tf.cos(np.pi * x[:, 1])
        elif(sys.argv[1] == '2'):
            return (x[:, 0] + 1) ** 2
        elif(sys.argv[1] == '3'):
            return 3 - tf.sin(2 * np.pi * x[:, 1])
        else:
            sys.exit("sys.argv[1] is error.")            
               
    def f(self, x):
        if(sys.argv[1] == '1'):
            return np.pi**2*(4*tf.sin(np.pi*x[:, 0])**2*tf.cos(np.pi*x[:, 1]) + 4*tf.sin(np.pi*x[:, 0]) - tf.cos(np.pi*x[:, 1]))*tf.sin(np.pi*x[:, 1])
        elif(sys.argv[1] == '2'):
            return 2*np.pi**2*x[:, 0]**5*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] - 1)) - 2*np.pi**2*x[:, 0]**5*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] + 1)) - 2*x[:, 0]**5 - 8*np.pi**2*x[:, 0]**4*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] + 1)) - 4*x[:, 0]**4 - 2*np.pi**2*x[:, 0]**3*x[:, 1]**2*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] - 1)) + 2*np.pi**2*x[:, 0]**3*x[:, 1]**2*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] + 1)) + 5*x[:, 0]**3*x[:, 1]**2 - 4*np.pi**2*x[:, 0]**3*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] - 1)) - 12*np.pi**2*x[:, 0]**3*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] + 1)) - 2*x[:, 0]**3 + 8*np.pi**2*x[:, 0]**2*x[:, 1]**2*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] - 1)) - 18*x[:, 0]**2*x[:, 1]**2 - 2*np.pi*x[:, 0]**2*x[:, 1]*tf.sin(2*np.pi*x[:, 1]*(x[:, 0] - 1)) + 2*np.pi*x[:, 0]**2*x[:, 1]*tf.sin(2*np.pi*x[:, 1]*(x[:, 0] + 1)) - 8*np.pi**2*x[:, 0]**2*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] + 1)) + 2*np.pi**2*x[:, 0]*x[:, 1]**4*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] - 1)) - 2*np.pi**2*x[:, 0]*x[:, 1]**4*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] + 1)) - 6*x[:, 0]*x[:, 1]**4 + 2*np.pi**2*x[:, 0]*x[:, 1]**2*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] - 1)) - 2*np.pi**2*x[:, 0]*x[:, 1]**2*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] + 1)) - 6*x[:, 0]*x[:, 1]**2 + 9*np.pi*x[:, 0]*x[:, 1]*tf.sin(2*np.pi*x[:, 1]*(x[:, 0] - 1)) - 3*np.pi*x[:, 0]*x[:, 1]*tf.sin(2*np.pi*x[:, 1]*(x[:, 0] + 1)) - x[:, 0]*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] - 1))/2 + 2*np.pi**2*x[:, 0]*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] - 1)) - 2*np.pi**2*x[:, 0]*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] + 1)) + x[:, 0]*tf.cos(2*np.pi*x[:, 1]*(x[:, 0] + 1))/2 + 2*np.pi*x[:, 1]**3*tf.sin(2*np.pi*x[:, 1]*(x[:, 0] - 1)) - 2*np.pi*x[:, 1]**3*tf.sin(2*np.pi*x[:, 1]*(x[:, 0] + 1)) + 2*np.pi*x[:, 1]*tf.sin(2*np.pi*x[:, 1]*(x[:, 0] - 1)) - 2*np.pi*x[:, 1]*tf.sin(2*np.pi*x[:, 1]*(x[:, 0] + 1)) - tf.cos(2*np.pi*x[:, 1]*(x[:, 0] - 1)) + tf.cos(2*np.pi*x[:, 1]*(x[:, 0] + 1))
        elif(sys.argv[1] == '3'):
            return x[:, 0]*tf.exp(x[:, 0]**2/2 + x[:, 1]**2/2) - x[:, 1]*tf.exp(x[:, 0]**2/2 + x[:, 1]**2/2) + 2*(-x[:, 0] + x[:, 1])*(x[:, 0]*x[:, 1]*tf.exp(x[:, 0]**2/2 + x[:, 1]**2/2) - 4*np.pi**2*tf.sin(2*np.pi*x[:, 1])*tf.sin(np.pi*(2*x[:, 0] + 2*x[:, 1])) + 4*np.pi**2*tf.cos(2*np.pi*x[:, 1])*tf.cos(np.pi*(2*x[:, 0] + 2*x[:, 1]))) + 2*np.pi*(x[:, 0]*tf.exp(x[:, 0]**2/2 + x[:, 1]**2/2) + 2*np.pi*tf.sin(2*np.pi*x[:, 1])*tf.cos(np.pi*(2*x[:, 0] + 2*x[:, 1])))*tf.sin(2*np.pi*x[:, 0]) + (tf.sin(2*np.pi*x[:, 1]) - 3)*(x[:, 1]**2*tf.exp(x[:, 0]**2/2 + x[:, 1]**2/2) + tf.exp(x[:, 0]**2/2 + x[:, 1]**2/2) - 8*np.pi**2*tf.sin(2*np.pi*x[:, 1])*tf.sin(np.pi*(2*x[:, 0] + 2*x[:, 1])) + 8*np.pi**2*tf.cos(2*np.pi*x[:, 1])*tf.cos(np.pi*(2*x[:, 0] + 2*x[:, 1]))) - (tf.cos(2*np.pi*x[:, 0]) + 3)*(x[:, 0]**2*tf.exp(x[:, 0]**2/2 + x[:, 1]**2/2) + tf.exp(x[:, 0]**2/2 + x[:, 1]**2/2) - 4*np.pi**2*tf.sin(2*np.pi*x[:, 1])*tf.sin(np.pi*(2*x[:, 0] + 2*x[:, 1]))) + 2*np.pi*(x[:, 1]*tf.exp(x[:, 0]**2/2 + x[:, 1]**2/2) + 2*np.pi*tf.sin(2*np.pi*x[:, 1])*tf.cos(np.pi*(2*x[:, 0] + 2*x[:, 1])) + 2*np.pi*tf.sin(np.pi*(2*x[:, 0] + 2*x[:, 1]))*tf.cos(2*np.pi*x[:, 1]))*tf.cos(2*np.pi*x[:, 1]) - 2*np.pi*tf.sin(np.pi*(2*x[:, 0] + 2*x[:, 1]))*tf.cos(2*np.pi*x[:, 1])
        else:
            sys.exit("sys.argv[1] is error.")    
        
    def g_d(self, x):
        return self.u(x)
        
    def g_n(self, x):
        gnlist = []
        if(sys.argv[1] == '1'):
            gnlist.append(-2*np.pi*tf.sin(np.pi*x[: self.batch_d, 1]))
            gnlist.append(-np.pi*(tf.sin(np.pi*x[self.batch_d: , 0]) + 2)*tf.sin(np.pi*x[self.batch_d: , 0]))
        elif(sys.argv[1] == '2'):
            return 0
        elif(sys.argv[1] == '3'):
            gnlist.append(x[: self.batch_d, 1]*(x[: self.batch_d, 1]*tf.exp(x[: self.batch_d, 1]**2/2) + 4*np.pi*tf.sin(2*np.pi*x[: self.batch_d, 1])*tf.cos(2*np.pi*x[: self.batch_d, 1])) - 8*np.pi*tf.sin(2*np.pi*x[: self.batch_d, 1])*tf.cos(2*np.pi*x[: self.batch_d, 1]))
            gnlist.append(-x[self.batch_d: , 0]**2*tf.exp(x[self.batch_d: , 0]**2/2) - 6*np.pi*tf.sin(2*np.pi*x[self.batch_d: , 0]))
        else:
            sys.exit("sys.argv[1] is error.")
        return tf.concat(gnlist, axis = 0)
            
    def norm2_grad(self, u, x):
        grad = tf.gradients(u, x)[0]
        return tf.reduce_sum(grad ** 2, axis = 1)

    def agradgrad(self, u, x):
        grad = tf.gradients(u, x)[0]
        return grad[:, 0] * (self.a11(x) * grad[:, 0] + self.a12(x) * grad[:, 1]) + grad[:, 1] * (self.a21(x) * grad[:, 0] + self.a22(x) * grad[:, 1])
        
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
        x_d = np.random.rand(2 * self.batch_d, self.dimension)
        x_n = np.random.rand(2 * self.batch_d, self.dimension)
        x_d[: self.batch_d, 0] = 1.
        x_d[self.batch_d: , 1] = 1.
        x_n[: self.batch_d, 0] = 0.
        x_n[self.batch_d: , 1] = 0.
        x_i = np.random.rand(self.batch, self.dimension)
        _, summary = sess.run([self.opt, self.merged], feed_dict = {self.x_d: x_d, self.x_n: x_n, self.x_i: x_i})
        return summary
            
    def test(self, sess, x):
        return sess.run([self.errl2, self.errh1, self.errh2], feed_dict = {self.x_i: x})

tf.compat.v1.disable_eager_execution()
net = resnet_ritz(2, 5, 10, "tanh") #dimension, layer, unit
with tf.compat.v1.Session() as sess:
    sess.run(net.init)
    train_writer = tf.compat.v1.summary.FileWriter('logs/rr', sess.graph)
    for i in range(500):
        train_writer.add_summary(net.train(sess), i)
    errl2, errh1, errh2 = net.test(sess, np.random.rand(10000, net.dimension))
print("d = {}, l = {}, u = {}".format(net.dimension, net.layer, net.unit))
net.net.summary()
print(sys.argv)
print("{:e} & {:e} & {:e}\\\\".format(errl2, errh1, errh2))
