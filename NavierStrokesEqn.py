import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import scipy.io

np.random.seed(1)
tf.random.set_random_seed(1)

N_train = 5000   
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

data = scipy.io.loadmat('C:/Users/Deiva/Downloads/cylinder_nektar_wake.mat')

U_star = data['U_star'] # N x 2 x T
P_star = data['p_star'] # N x T
t_star = data['t'] # T x 1
X_star = data['X_star'] # N x 2
N = X_star.shape[0]
T = t_star.shape[0]



x = np.tile(X_star[:,0:1], (1,T)) # N x T
y = np.tile(X_star[:,1:2], (1,T)) # N x T
t = np.tile(t_star, (1,N)).T
u = U_star[:,0,:] # N x T
v = U_star[:,1,:] # N x T
p = P_star # N x T

x = x.flatten()[:,None] # NT x 1
y = y.flatten()[:,None] # NT x 1
t = t.flatten()[:,None] # NT x 1
u = t.flatten()[:,None] # NT x 1
v = t.flatten()[:,None] # NT x 1
p = p.flatten()[:,None] # NT x 1

idx = np.random.choice(N*T, N_train, replace=False)
x_train = x[idx,:]
y_train = y[idx,:]
t_train = t[idx,:]
u_train = u[idx,:]
v_train = v[idx,:]

class PDENN:
  def __init__(self,x,y,t,u,v,layers):

    X = np.concatenate([x,y,t],1)

    self.ub = X.max(0)
    self.lb = X.min(0)
    self.X = X
    self.x = X[:,0:1]
    self.y = X[:,1:2]
    self.t = X[:,2:3]
    self.u = u
    self.v = v
    self.layers = layers
    self.weights, self.biases = self.initialize_NN(layers)

    self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
    self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)

    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
    self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
    self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
    self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
    self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])

    self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf, self.t_tf)

    self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + tf.reduce_sum(tf.square(self.f_u_pred)) + tf.reduce_sum(tf.square(self.f_v_pred))
    
    self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', options = {'maxiter': 50000, 'maxfun': 50000, 'maxcor': 50, 'maxls': 20, 'ftol' : 1.0 * np.finfo(float).eps})

    self.optimizer_Adam = tf.train.AdamOptimizer()
    self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
    init = tf.global_variables_initializer()
    self.sess.run(init)


  def initialize_NN(self, layers):

    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0, num_layers-1):
      W = self.xavier_init(size = [layers[l], layers[l+1]])
      b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32) 
      weights.append(W)
      biases.append(b)
    return weights, biases

  def xavier_init(self, size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

  def net_NS(self, x, y, t):

    lambda_1 = self.lambda_1
    lambda_2 = self.lambda_2
        
    psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
    psi = psi_and_p[:,0:1]
    p = psi_and_p[:,1:2]
        
    u = tf.gradients(psi, y)[0]
    v = -tf.gradients(psi, x)[0]  
        
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
        
    v_t = tf.gradients(v, t)[0]
    v_x = tf.gradients(v, x)[0]
    v_y = tf.gradients(v, y)[0]
    v_xx = tf.gradients(v_x, x)[0]
    v_yy = tf.gradients(v_y, y)[0]
        
    p_x = tf.gradients(p, x)[0]
    p_y = tf.gradients(p, y)[0]

    f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
    f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)

    return u, v, p, f_u, f_v

  def neural_net(self, X, weights, biases):

    num_layers = len(weights) + 1
        
    H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
    for l in range(0,num_layers-2):
      
      W = weights[l]
      b = biases[l]
      H = tf.tanh(tf.add(tf.matmul(H, W), b))

    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

  def train(self, nIter): 

    tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,self.u_tf: self.u, self.v_tf: self.v}
        
    start_time = time.time()
    pll=[]
    i=0
    pllin=[]
    for it in range(nIter):
      self.sess.run(self.train_op_Adam, tf_dict)
     # Print
      if it % 1000 == 0:
        elapsed = time.time() - start_time
        loss_value = self.sess.run(self.loss, tf_dict)
        pll.append(loss_value)
        pllin.append(it)
        lambda_1_value = self.sess.run(self.lambda_1)
        lambda_2_value = self.sess.run(self.lambda_2)
        print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
        start_time = time.time()
    self.optimizer.minimize(self.sess, feed_dict = tf_dict, fetches = [self.loss, self.lambda_1, self.lambda_2], loss_callback = self.callback)
    plt.figure() 
    plt.title('Loss Function')
    plt.xlabel('nth Iteration')
    plt.ylabel('Loss Value')
    plt.plot(pllin,pll,'r-',)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
        

  def callback(self, loss, lambda_1, lambda_2):
    print('Loss: %.3e, l1: %.3f, l2: %.5f' % (loss, lambda_1, lambda_2))
    
model = PDENN(x_train, y_train, t_train, u_train, v_train, layers)
model.train(50000)
