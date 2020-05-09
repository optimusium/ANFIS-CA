import os
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
## from tensorflow.keras.utils import plot_model
import tensorflow.contrib.slim as slim
from anfis_GAU import ANFIS # 3 ANFIS classes to be coded later i.e., BELL, TRIANGLE, & TRAPEZOIDAL

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def train_testData(dataset="Data.csv", n = 4):
    Data = pd.read_csv(dataset)
    y = Data.Action
    x = Data.drop("Action", axis=1)
    print("Shape of x: " + str(x.shape) + "\n")
    trnData, chkData, trnLbls, chkLbls = train_test_split(x, y, test_size=0.2)

    trnData = trnData.to_numpy()
    chkData = chkData.to_numpy()
    trnLbls = trnLbls.to_numpy()
    chkLbls = chkLbls.to_numpy()
    
    trn = {}
    for i in range(n):
        trn["trnDataF{}".format(i)] = trnData[:, i]
        
    chk = {}
    for i in range(n):
        chk["chkDataF{}".format(i)] = chkData[:, i]
        
    bias = {}
    bias["bias_trn"] = np.ones(trnLbls.shape)
    bias["bias_chk"] = np.ones(chkLbls.shape)
    
    return trn, chk, trnLbls, chkLbls, bias

def trainingNplotting(fis=None, trn=None, trnLbls=None, chk=None,
                      chkLbls=None, bias=None, epochs=2000):
    num_epochs = epochs
    # Initialize session to make computations on the Tensorflow graph
    with tf.Session() as sess:
        # Visualization of the model in TensorBoard
        if not os.path.exists('summaries'):
            os.mkdir('summaries')
        if not os.path.exists(os.path.join('summaries','first')):
            os.mkdir(os.path.join('summaries','first'))
        summ_writer = tf.summary.FileWriter(os.path.join('summaries','first'), sess.graph)
        # Initialize model parameters
        sess.run(fis.init_variables)
        
        trn_costs = []
        val_costs = []
        
        trn_predList = []
        sigValSUBtrnLbls = []
        sigList = []
        
        
        avg_loss = []
        avg_test_accuracy = []
        
        accuracyList = []
        
        time_start = time.time()
        for epoch in range(num_epochs):
            # Run an update step
            # if epoch == 0:
                # val_pred, val_loss, gn_summ = fis.firstEpoch(sess, trnData, trnLbls, tf_gradnorm_summary_y)
                # summ_writer.add_summary(gn_summ, epoch)
            trn_pred, trn_loss = fis.train(sess, trn, trnLbls, bias)
    
            # sigVal = sig >= 0.5
            
            # if sig >= 0.5:
            #     sigVal = 1
            # else:
            #     sigVal = 0
            # Evaluate on validation set
            val_pred, val_loss = fis.infer(sess, chk, chkLbls, bias)
            if epoch % 100 == 0:
                print("Train cost after epoch %i: %f" % (epoch, trn_loss))
                # print("trn_pred: " + str(trn_pred))
                # print("trnLbls: " + str(trnLbls))
                # print("trn_pred - trnLbls: " + str(trn_pred - trnLbls))
            if epoch == num_epochs - 1:
                time_end = time.time()
                print("\n" + "Elapsed time: %f" % (time_end - time_start))
                print("Validation loss: %f" % val_loss)
                # plt.figure(1)
                # plt.plot(sigVal - trnLbls)
                # print(trnLbls)
                # print(trnLbls.shape)
                # print(trn_pred.shape)
                
            # acc = sum(sigVal - trnLbls == 0)

            trn_costs.append(trn_loss)
            val_costs.append(val_loss)
            trn_predList.append(trn_pred)
            # sigValSUBtrnLbls.append(sigVal - trnLbls)
            avg_loss.append(np.mean(trn_loss))
            avg_test_accuracy.append(np.mean(val_loss))
            
            # sigList.append(sigVal)
            
            # accuracyList.append(acc / len(trnLbls))
            
            # plt.figure(1)
            # plt.plot(accuracyList)
            
            
        

        
        # Plot the cost over epochs
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(np.squeeze(trn_costs))
        plt.title("Training loss, Learning rate =" + str(alpha))
        plt.subplot(2, 1, 2)
        plt.plot(np.squeeze(val_costs))
        plt.title("Validation loss, Learning rate =" + str(alpha))
        plt.ylabel('Cost')
        plt.xlabel('Epochs')
        # Plot resulting membership functions
        
        # rule_stats, mus, sigmas, y = fis.plotmfs(sess)
        
        # # Execute the summaries defined above
        # # summ = sess.run(performance_summaries, feed_dict={tf_loss_ph:avg_loss, tf_accuracy_ph:accuracyList})
        
        # with tf.name_scope('performance'):
        #     summary = tf.Summary(value=[tf.Summary.Value(tag="trn_loss", simple_value=trn_loss)])

        # summ_writer.add_summary(summary)
        

        # summ_writer.add_summary(tf.summary("accuracy", tf.constant(accuracyList)), epoch)
        
        # print("summ: " + str(type(summ)))
        # Write the obtained summaries to the file, so it can be displayed in the TensorBoard
        # summ_writer.add_summary(summ, epoch)
        
    # inp_list = ["Seniority ", "Propensity ","Size of Company ", "Contactibility "]
    # action = "Send Brochure & Product Updates"
    # for r in rule_stats:
    #     if abs(rule_stats[r]["center"]) > 0.05:
    #         term = "\nrule %s: IF(" % r
    #         for inp in range(4):
    #             hb = rule_stats[r][inp]["high_bound"]
    #             lb = rule_stats[r][inp]["low_bound"]
    #             significance = ""
                
    #             if lb >= 1 or hb <= 0: 
    #                 significance = "is not related "
    #             elif hb == 1 and lb == 0: 
    #                 significance = "is in any value "
    #             else:
    #                 nothing = 0
    #                 if lb == 0:
    #                     nothing = 1 # extremely low
                    
    #                 extreme = 0
    #                 if hb == 1:
    #                     extreme = 1 # extremely high
                    
    #                 some_little = 0
    #                 if hb < 0.34:
    #                     some_little = 1 # in lowest one-third 
                    
    #                 low = 0
    #                 if hb < 0.49:
    #                     low = 1 # highest value is just below half
                    
    #                 some_high = 0
    #                 if lb > 0.66:
    #                     some_high = 1 # in highest one-third
                    
    #                 high = 0
    #                 if lb > 0.51:
    #                     high = 1 # lowest value is just above half
                    
    #                 # some_medium = 0
    #                 # if lb > 0.33 and lb < 0.66:
    #                     # some_medium = 1 # lowest value is just above lowest one-third
                    
    #                 # if hb > 0.33 and hb < 0.66:
    #                     # some_medium = 1 # highest value is just below highest one-third

    #                 if nothing == 1:
    #                     if low == 1:
    #                         significance = "is low " # highest value is just below half
    #                     elif some_little == 1:
    #                         significance = "is very low " # in lowest one-third
    #                     elif hb > 0.49:
    #                         significance = "is not extreme high " # large spread with highest value that may be close to extremely high
    #                     else:
    #                         significance = "is nothing "
                    
    #                 elif extreme == 1:
    #                     if high == 1:
    #                         significance = "is high " # lowest value is just above half
    #                     elif some_high == 1:
    #                         significance = "is very high " # in highest one-third
    #                     elif lb < 0.51:
    #                         significance = "is not extremely low " # large spread with lowest value that may be close to extremely low
    #                     else:
    #                         significance = "is extremely high "
                    
    #                 elif low == 1 and some_little == 1:
    #                     significance = "is low but not zero " # in lowest one-third
    #                 elif low == 1:
    #                     significance = "is medium low " # # highest value is just below half
    #                 elif some_little == 1:
    #                     significance = "is very low but not zero " # in lowest one-third
                    
    #                 elif high == 1 and some_high == 1:
    #                     significance = "is high but not zero " # in highest one-third
    #                 elif high == 1:
    #                     significance = "is medium high " # lowest value is just above half
    #                 elif some_high == 1:
    #                     significance = "is quite high but not extermely high " # in highest one-third
                   
    #                 elif lb > 0.33 and lb < 0.66 and hb < 0.66: 
    #                     significance = "is medium " # in the middle one-third 
    #                 elif lb > 0.33 and lb < 0.66 and hb < 1:
    #                     significance = "is medium and high " # in highest two-third 
                   
    #                 elif hb > 0.33 and hb < 0.66 and lb > 0.33:
    #                     significance = "is medium " # in the middle one-third
    #                 elif hb > 0.33 and hb < 0.66 and lb > 0:
    #                     significance = "is medium and low " # in lowest two-third

    #             term += inp_list[inp] + significance + "AND "
    #         term = term[:-4]
    #         if rule_stats[r]["center"] >= 0.5:
    #             term+=") THEN " +action+" (%s)" % rule_stats[r]["center"]
    #         else:
    #             term+=") THEN "+action+" (%s)" % rule_stats[r]["center"]
    #         print(str(term) + "\n")
                    
    plt.show()
    
    # return trn_predList, avg_loss, avg_test_accuracy, sigValSUBtrnLbls,sigList, accuracyList, wifi, mus, sigmas, y          
    
    return trn_pred, trn_loss, val_pred, val_loss


# ANFIS params and Tensorflow graph initialization
n = 4  # number of inputs
m = 81  # number of rules
f = 3 # number of linguistic labels
alpha = 0.0009  # learning rate, that will be tune by GA to be coded later

tf.reset_default_graph()

fis = ANFIS(n_inputs=n, n_rules=m, n_fuzzy=f, learning_rate=alpha)

# tf_learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')
# optimizer = tf.train.MomentumOptimizer(alpha, momentum=0.9)
# grads_and_vars = optimizer.compute_gradients(fis.loss)

# Name scope allows you to group various summaries together
# Summaries having the same name_scope will be displayed on the same row
# with tf.name_scope('performance'):
#     # Summaries need to be displayed
#     # Whenever you need to record the loss, feed the mean loss to this placeholder
#     tf_loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
#     # Create a scalar summary object for the loss so it can be displayed
#     tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)

#     # Whenever you need to record the loss, feed the mean test accuracy to this placeholder
#     tf_accuracy_ph = tf.placeholder(tf.float32, shape=None, name='accuracy_summary')
#     # Create a scalar summary object for the accuracy so it can be displayed
#     tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)

# Gradient norm summary
# for g, v in grads_and_vars:
    
#     print("g: " + str(g))
#     print("v: " + str(v))
    
#     with tf.name_scope('gradients'):
#         if 'mu' in v.name:
#             tf_last_grad_norm_mu = tf.sqrt(tf.reduce_mean(g**2))
#             tf_gradnorm_summary_mu = tf.summary.scalar('grad_norm_mu', tf_last_grad_norm_mu)
#         if 'sigma' in v.name:
#             tf_last_grad_norm_sigma = tf.sqrt(tf.reduce_mean(g**2))
#             tf_gradnorm_summary_sigma = tf.summary.scalar('grad_norm_sigma', tf_last_grad_norm_sigma)
#         if 'y' in v.name:
#             tf_last_grad_norm_y = tf.sqrt(tf.reduce_mean(g**2))
#             tf_gradnorm_summary_y = tf.summary.scalar('grad_norm_y', tf_last_grad_norm_y)
            
            # break
# Merge all summaries together
# performance_summaries = tf.summary.merge([tf_loss_summary, tf_accuracy_summary])

model_summary()

# There are 3 actions that correspond to the 2 sets of output values
# (i) Action1 vs. Action2 or Action3
# Discard Contact when both output values are negative in the following 2 sections

# Action2 vs. Action1 or Action3 
# Send Brochure & Product Updates when output values are positive in this section

# Training, this will be coded later in the train.py
trn, chk, trnLbls, chkLbls, bias = train_testData("Data.csv", n)

# Action3 vs. Action1 or Action2
# Send Email & Mark Contact for Call when output values are positive in this section

# Training, this will be coded later in the train.py
# trnData, chkData, trnLbls, chkLbls = train_testData("Data2.csv")


trn_pred, trn_loss, val_pred, val_loss = trainingNplotting(fis, trn, trnLbls, chk, chkLbls, bias)










# trn_predList, avg_loss, avg_test_accuracy, sigValSUBtrnLbls, sigList, accuracyList, wifi, mus, sigmas, y = trainingNplotting()




