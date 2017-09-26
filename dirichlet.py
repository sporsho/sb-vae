from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

#this code is copied and adopted from https://github.com/blei-lab/edward/blob/master/examples/pp_dirichlet_process.py
def dirichlet_process(alpha):
    """dirichlet process with stick breaking process"""
    def cond(k, beta_k):
        if k==10:
            return True
        else:
            return False
    def body(k, beta_k):
        beta_k*=tf.distributions.Beta(1.0, alpha).sample((5,5))
        return k+1, beta_k
    k= tf.constant(0)
    beta_k=tf.distributions.Beta(1.0, alpha).sample((5,5))
    stick_num, stick_beta= tf.while_loop(cond, body, loop_vars=[0, beta_k])
    return stick_beta

dp= dirichlet_process(5.0)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(dp))
    
    