### training script for Stick Breaking VAE ###

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from SBVAE_Model import SBVAE_Model

np.random.seed(0)
tf.set_random_seed(0)

#get data 
mnist= input_data.read_data_sets("MNIST_data/", one_hot=True)
#configuration
# for simplicity we are considering a inference model with 2 hidden layer and a generative model with 2 hidden layer
n_epochs=100
n_batch=100
n_latent=20
learning_rate=0.001
n_samples=mnist.train.num_examples
network_arch=dict(n_hidden_encoder_1=500,
                  n_hidden_encoder_2=500,
                  n_hidden_generative_1=500,
                  n_hidden_generative_2=500,
                  n_input=784, # 28*28 images
                  n_z=20)


#plt.figure(figsize=(4,4))
#plt.imshow(batch_xs[0].reshape(28,28), vmin=0, vmax=1, cmap="gray")
#plt.show()
def train(network_arch, learning_rate=0.001, batch_size=100, training_epochs=10):
    print str(training_epochs)+" "+str(n_samples)
    vae= SBVAE_Model(network_arch, tf.nn.softplus)
    #training each n_epochs
    for epoch in range(training_epochs):
        avg_cost=0.
        total_batch= int(n_samples/batch_size)
        # go through all the batches of data
        for i in range(total_batch):
            batch_data,_=mnist.train.next_batch(batch_size)
            #fit training data
            cost= vae.fit_data(batch_data)
            #K, a, b= vae.get_K(batch_data)
            #print(K.shape)
            avg_cost+=cost/n_samples*batch_size
        #display log per 5 epoch
        if epoch % 5==0:
            print("Epoch: ", '%04d'% (epoch+1), "cost= ", "{:.9f}".format(avg_cost))
    return vae


sbvae= train(network_arch, training_epochs=50)
x_sample= mnist.test.next_batch(100)[0]
x_reconstruct= sbvae.reconstruct_data(x_sample)
plt.figure(figsize=(8,12))
for i in range(5):
    print("figure")
    plt.subplot(5,2,2*i+1)
    plt.imshow(x_sample[i].reshape(28,28), vmin=0, vmax=1, cmap='gray')
    plt.title('test input')
    plt.colorbar()
    plt.subplot(5,2,2*i+2)
    plt.imshow(x_reconstruct[i].reshape(28,28), vmin=0, vmax=1, cmap='gray')
    plt.title('reconstruction')
    plt.colorbar()
plt.tight_layout()
plt.show()
