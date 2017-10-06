### Stick breaking VAE Model ###
### No Rights reserved. No guarrantee is provided ###
### Reference: https://arxiv.org/pdf/1605.06197.pdf
### Part of Masters Practikum ###
### Deep Learning for Real World ###
### Faculty of Informatics ###
### Technical University of Munich ###


###The stick breaking dirichlet process is adapted from Edward's example###
### link: https://github.com/blei-lab/edward/blob/master/examples/pp_dirichlet_process.py ###
### Initially I planned to implement the kumaraswamy process with stick breking, but it is a bit complex for ###
### multilayer neural network. The original author of  SBVAE implemented kumaraswamy process with ###
### stick breaking using a Single Layer network which makes the computation of 'a' and 'b' easy. ###
### this code is much faster and accurrate than normal VAE with gaussian distribution (rather than ###
### dirichlet process with stick breaking prior) but this code suffers from Vanishing Gradient problem ###
### Adam Optimizer is used for Machine Learning, other optimizer are not tested yet. ###

### This code is written with the help from stackoverflow, the original author's theano implementation ###
### and Jan Hendrik Metzen's personal blog. ###

import numpy as np
import tensorflow as tf
from xavier import xavier_init

class SBVAE_Model(object):
    def __init__(self, network_arch, activation_func=tf.nn.softplus, learning_rate=0.001, batch_size=100):
        self.network_arch= network_arch
        self.activation_func= activation_func
        self.learning_rate= learning_rate
        self.batch_size= batch_size
        
        #input graph
        self.X= tf.placeholder(tf.float32, [None, network_arch["n_input"]])
        
        #create vae network
        self._create_network()
        # initialize tf
        init= tf.global_variables_initializer()
        # launch the session
        self.sess= tf.InteractiveSession()
        self.sess.run(init)
        self.param=[self.network_arch, self.activation_func, self.learning_rate, self.X]
        # little known kumaraswamy samples
    def _get_kumaraswamy_sample(self, batch_size, latent_size, weights1, weights2):
        uniform_samples= tf.random_uniform(shape=(batch_size, latent_size), minval=0.01, maxval=0.99, dtype=tf.float32)
        self.a= tf.nn.softplus(tf.tensordot(self.X, weights1['out_mean'],1))
        self.b= tf.nn.softplus(tf.tensordot(self.X, tf.transpose(weights2['out_mean']),1))
        #ks=(1-(uniform_samples**(1/self.b)))**(1/self.a)
        ks=tf.pow((1.-tf.pow(uniform_samples, (1./self.b))),(1./self.a))
        return ks
        
        
    def dirichlet_process(self, alpha, latent_size, beta_k):
        """dirichlet process with stick breaking process"""
        def cond(k, beta_k):
            if k==latent_size:
                return True
            else:
                return False
        def body(k, beta_k):
            beta_k*=tf.distributions.Beta(1.0, alpha).sample((self.batch_size,latent_size))
            return k+1, beta_k
        k= tf.constant(0)
        #beta_k=tf.distributions.Beta(1.0, alpha).sample((self.batch_size, latent_size))
        stick_num, stick_beta= tf.while_loop(cond, body, loop_vars=[0, beta_k])
        return stick_beta


    #create network for VAE
    def _create_network(self):
        network_weights=self._initialize_weights(**self.network_arch)
        network_biases= self._initialize_biases(**self.network_arch)
        self.z_mean, self.z_log_sigma_sq= self._encoder_network(network_weights["encoder"], network_biases["encoder"])
        
        #for normal VAE draw one sample z from Gaussian Dist
        n_z= self.network_arch["n_z"]
        #eps=tf.random_normal((self.batch_size, n_z), 0,1, dtype=tf.float32)
        
        #z= mu+sigma* epsilon
        #self.Z= tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)),eps))
        
        # for SB VAE draw samples from stick breaking process with kumaraswamy samples
        #self.kumaraswamy= self._get_kumaraswamy_sample(self.batch_size, n_z, network_weights["encoder"], network_weights["generative"])
        #kumaraswamy is not stable now, beta distribution is similar to it, so we will use it in the meantime
        #self.K= self._get_kumaraswamy_sample(self.batch_size, n_z, network_weights["encoder"], network_weights["generative"])
        # now break the stick n_z times to get GEM distribution
        #stick_segment= tf.Variable(np.zeros((self.batch_size,)), dtype=tf.float32, name="stick_segment")
        #remaining_segment=tf.Variable(np.ones((self.batch_size,)), dtype=tf.float32, name="remaining_segment")
        
        #for m in range(n_z):
          #  stick_segment=self.kumaraswamy[:,m]*remaining_segment
            #remaining_segment*=(1-self.kumaraswamy[:,m])
        
        
        #print(self.z_mean.shape)
        #now comes the latent variable
        self.Z= self.dirichlet_process(1.0, n_z, self.z_mean)
        
        #use generator to determine mean of Bernoulli distribution of reconstructed inputs
        self.x_reconstruction_mean= self._generative_network(network_weights["generative"], network_biases["generative"])
        
        #loss and optimizer
        
        reconstruction_loss= -tf.reduce_sum(self.X*tf.log(1e-10+self.x_reconstruction_mean)+(1-self.X)*tf.log(1e-10+1-self.x_reconstruction_mean), 1)
        #KL divergence is the latent loss
        latent_loss= -0.5*tf.reduce_sum(1+self.z_log_sigma_sq-tf.square(self.z_mean)-tf.exp(self.z_log_sigma_sq),1)
        self.cost= tf.reduce_mean(reconstruction_loss+ latent_loss)
        
        self.optimizer= tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def _encoder_network(self, weights, biases):
        #generate probabilistic encoder which maps inputs onto a stick breaking GEM distribution in latent space
        layer_1=self.activation_func(tf.add(tf.matmul(self.X, weights['h1']), biases['b1']))
        layer_2= self.activation_func(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        z_mean= tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        z_log_sigma_sq=tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)
    
    def _generative_network(self, weights, biases):
        #decoder network which maps points in latent space into a Bernoulli Distribution in Data Space
        layer_1= self.activation_func(tf.add(tf.matmul(self.Z, weights['h1']),biases['b1']))
        layer_2= self.activation_func(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        x_reconstruction_mean= tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean']))
        return x_reconstruction_mean
        
    def _initialize_weights(self, n_hidden_encoder_1, n_hidden_encoder_2, n_hidden_generative_1, n_hidden_generative_2, n_input, n_z):
        all_weights= dict()
        all_weights["encoder"]={
            'h1': tf.Variable(xavier_init(n_input, n_hidden_encoder_1)),
            'h2': tf.Variable(xavier_init(n_hidden_encoder_1, n_hidden_encoder_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_encoder_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_encoder_2, n_z))}
        all_weights["generative"]={
            'h1': tf.Variable(xavier_init(n_z, n_hidden_generative_1)),
            'h2': tf.Variable(xavier_init(n_hidden_generative_1, n_hidden_generative_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_generative_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_generative_2, n_input))}
        
        return all_weights
    def _initialize_biases(self, n_hidden_encoder_1, n_hidden_encoder_2, n_hidden_generative_1, n_hidden_generative_2, n_input, n_z):
        all_biases= dict()
        all_biases["encoder"]={
            'b1': tf.Variable(tf.zeros([n_hidden_encoder_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_encoder_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_biases["generative"]={
            'b1': tf.Variable(tf.zeros([n_hidden_generative_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_generative_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype= tf.float32))}
        return all_biases
    def get_K(self, X):
        return self.sess.run((self.K, self.a, self.b), feed_dict={self.X:X})
        
    def fit_data(self,X):
        #train data based on mini batches
        opt, cost= self.sess.run((self.optimizer, self.cost), feed_dict={self.X:X})
        return cost
    def map_to_latent_space(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.X:X})
    def generate_data(self, z_mu=None):
        # generate data by sampling from latent space
        if z_mu is None:
            z_mu=np.random_normal(size=self.network_arch["n_z"])
        return self.sess.run(self.x_reconstruction_mean, feed_dict={self.Z:z_mu})
    def reconstruct_data(self, X):
        return self.sess.run(self.x_reconstruction_mean, feed_dict={self.X:X})
        