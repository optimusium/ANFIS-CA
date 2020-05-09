import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ANFIS:

    def __init__(self, n_inputs, n_rules, learning_rate=1e-2):
        self.n = n_inputs
        self.m = n_rules
        self.inputs = tf.placeholder(tf.float32, shape=(None, n_inputs))  # Input
        self.targets = tf.placeholder(tf.float32, shape=None)  # Desired output
        mu = tf.get_variable("mu", [n_rules * n_inputs],
                             initializer=tf.random_normal_initializer(0, 1), constraint=lambda t: tf.clip_by_value(t, 0.15, 0.85))  # Means of Gaussian MFS
        sigma = tf.get_variable("sigma", [n_rules * n_inputs],
                                initializer=tf.random_normal_initializer(0, 1), constraint=lambda t: tf.clip_by_value(t, -0.33, 0.33))  # Standard deviations of Gaussian MFS
        y = tf.get_variable("y", [1, n_rules], initializer=tf.random_normal_initializer(-1, 1), constraint=lambda t: tf.clip_by_value(t, -1, 1))  # Sequent centers
        #y = tf.clip_by_value(y,-1,1)
        self.params = tf.trainable_variables()

        self.rul = tf.reduce_prod(
            tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.inputs, (1, n_rules)), mu)) / tf.square(sigma)),
                       (-1, n_rules, n_inputs)), axis=2)  # Rule activations
        # Fuzzy base expansion function:
        num = tf.reduce_sum(tf.multiply(self.rul, y), axis=1)
        den = tf.clip_by_value(tf.reduce_sum(self.rul, axis=1), 1e-12, 1e12)
        self.out = tf.divide(num, den)

        self.loss = tf.losses.huber_loss(self.targets, self.out)  # Loss function computation
        #self.loss = tf.losses.absolute_difference(self.targets, self.out)
        # Other loss functions for regression, uncomment to try them:
        # loss = tf.sqrt(tf.losses.mean_squared_error(target, out))
        # loss = tf.losses.absolute_difference(target, out)
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)  # Optimization step
        # Other optimizers, uncomment to try them:
        #self.optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss)
        # self.optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.init_variables = tf.global_variables_initializer()  # Variable initializer

    def infer(self, sess, x, targets=None):
        if targets is None:
            return sess.run(self.out, feed_dict={self.inputs: x})
        else:
            return sess.run([self.out, self.loss], feed_dict={self.inputs: x, self.targets: targets})

    def train(self, sess, x, targets):
        yp, l, _ = sess.run([self.out, self.loss, self.optimize], feed_dict={self.inputs: x, self.targets: targets})
        return l, yp

    def plotmfs(self, sess):
        rule_stats={}
        mus = sess.run(self.params[0])
        mus = np.reshape(mus, (self.m, self.n))
        sigmas = sess.run(self.params[1])
        sigmas = np.reshape(sigmas, (self.m, self.n))
        y = sess.run(self.params[2])
        xn = np.linspace(0, 1, 1000)
        for r in range(self.m):
            rule_stats[r+1]={}
            if r % 4 == 0:
                plt.figure(figsize=(11, 6), dpi=80)
            plt.subplot(2, 2, (r % 4) + 1)
            ax = plt.subplot(2, 2, (r % 4) + 1)
            ax.set_title("Rule %d, sequent center: %f" % ((r + 1), y[0, r]))
            print(np.sum(y))
            rule_stats[r+1]["center"]=y[0, r] #/np.sum(y)
            for i in range(self.n):
                rule_stats[r+1][i]={}
                #plt.legend(loc="upper right")
                low_bound=mus[r, i]-abs(sigmas[r, i])
                if low_bound<0: low_bound=0
                if low_bound>1: low_bound=1
                high_bound=mus[r, i]+abs(sigmas[r, i])
                if high_bound>1: high_bound=1
                if high_bound<0: high_bound=0
                rule_stats[r+1][i]["high_bound"]=high_bound
                rule_stats[r+1][i]["low_bound"]=low_bound
                
                plt.plot(xn, np.exp(-0.5 * ((xn - mus[r, i]) ** 2) / (sigmas[r, i] ** 2)) ,"C%s" % i, label="%i" % i)
                print("rule %s input %i: mu=%s sigma=%s bound=[%s,%s]" %(r+1,i,mus[r, i],sigmas[r, i], low_bound, high_bound    ))
                
        return rule_stats
                
