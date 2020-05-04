import pandas as pd
from sklearn.model_selection import train_test_split

#Action2
Data=pd.read_csv("Data.csv")
y=Data.Action
x=Data.drop("Action",axis=1)
print(x.shape)
trnData, chkData, trnLbls, chkLbls = train_test_split(x, y, test_size=0.2)

trnData=trnData.to_numpy()
chkData=chkData.to_numpy()
trnLbls=trnLbls.to_numpy()
chkLbls=chkLbls.to_numpy()

#%load_ext tensorboard
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from anfis import ANFIS
from tensorflow.keras.utils import plot_model
import tensorflow.contrib.slim as slim

'''
# Mackey-Glass series computation
def mackey(n_iters):
    x = np.zeros((n_iters,))
    x[0:30] = 0.23 * np.ones((30,))
    t_s = 30
    for i in range(30, n_iters - 1):
        a = x[i]
        b = x[i - t_s]
        y = ((0.2 * b) / (1 + b ** 10)) + 0.9 * a
        x[i + 1] = y
    return x
'''

# Generate dataset
D = 4  # number of regressors
T = 1  # delay
N = 2000  # Number of points to generate
'''
mg_series = mackey(N)[499:]  # Use last 1500 points
data = np.zeros((N - 500 - T - (D - 1) * T, D))
lbls = np.zeros((N - 500 - T - (D - 1) * T,))

for t in range((D - 1) * T, N - 500 - T):
    data[t - (D - 1) * T, :] = [mg_series[t - 3 * T], mg_series[t - 2 * T], mg_series[t - T], mg_series[t]]
    lbls[t - (D - 1) * T] = mg_series[t + T]
trnData = data[:lbls.size - round(lbls.size * 0.3), :]
trnLbls = lbls[:lbls.size - round(lbls.size * 0.3)]
chkData = data[lbls.size - round(lbls.size * 0.3):, :]
chkLbls = lbls[lbls.size - round(lbls.size * 0.3):]
'''
# ANFIS params and Tensorflow graph initialization
m = 16  # number of rules
alpha = 0.0009  # learning rate

fis = ANFIS(n_inputs=D, n_rules=m, learning_rate=alpha)
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

model_summary()
#%tensorboard --logdir logs


# Action2
# Training
num_epochs = 6000

# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    # Initialize model parameters
    sess.run(fis.init_variables)
    trn_costs = []
    val_costs = []
    time_start = time.time()
    for epoch in range(num_epochs):
        #  Run an update step
        trn_loss, trn_pred = fis.train(sess, trnData, trnLbls)
        # Evaluate on validation set
        val_pred, val_loss = fis.infer(sess, chkData, chkLbls)
        if epoch % 10 == 0:
            print("Train cost after epoch %i: %f" % (epoch, trn_loss))
        if epoch == num_epochs - 1:
            time_end = time.time()
            print("Elapsed time: %f" % (time_end - time_start))
            print("Validation loss: %f" % val_loss)
            # Plot real vs. predicted
            pred = np.vstack((np.expand_dims(trn_pred, 1), np.expand_dims(val_pred, 1)))
            plt.figure(1)
            plt.plot(trn_pred-trnLbls)
            print(trnLbls)
            print(trnLbls.shape)
            #plt.plot(trn_pred)
            print(trn_pred.shape)
        trn_costs.append(trn_loss)
        val_costs.append(val_loss)
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
    rule_stats=fis.plotmfs(sess)
    inp_list=["Seniority ", "Propensity ","Size of Company ", "Contactibility "]
    action="send brochure"
    for r in rule_stats:
        if abs(rule_stats[r]["center"])>0.05:
            #print("rule %s" % r)
            term="\nrule %s: IF(" % r
            for inp in range(4):
                hb= rule_stats[r][inp]["high_bound"]
                lb= rule_stats[r][inp]["low_bound"]
                significance=""
                if lb>=1 or hb<=0: significance="is not related "
                elif hb==1 and lb==0: significance="is in any value "
                else:
                    nothing=0
                    if lb==0:
                        nothing=1
                    extreme=0
                    if hb==1:
                        extreme=1
                    some_little=0
                    if hb<0.34:
                        some_little=1
                    low=0
                    if hb<0.49:
                        low=1
                    some_high=0
                    if lb>0.66:
                        some_high=1
                    high=0
                    if lb>0.51:
                        high=1
                    some_medium=0
                    if lb>0.33 and lb<0.66:
                        some_medium=1
                    if hb>0.33 and hb<0.66:
                        some_medium=1

                    if nothing==1:
                        if low==1:
                            significance="is low "
                        elif some_little==1:
                            significance="is very low "
                        elif hb>0.49:
                            significance="is not extreme high "
                        else:
                            significance="is nothing "
                    elif extreme==1:
                        if high==1:
                            significance="is high "
                        elif some_high==1:
                            significance="is very high "
                        elif lb<0.51:
                            significance="is not extremely low "
                        else:
                            significance="is extremely high "
                    elif low==1 and some_little==1:
                        significance="is low but not zero "
                    elif low==1:
                        significance="is medium low "
                    elif some_little==1:
                        significance="is very low but not zero "
                    elif high==1 and some_high==1:
                        significance="is high but not zero "
                    elif high==1:
                        significance="is medium high "
                    elif some_high==1:
                        significance="is quite high but not extermely high "
                    elif lb>0.33 and lb<0.66 and hb<0.66:
                        significance="is medium "
                    elif lb>0.33 and lb<0.66 and hb<1:
                        significance="is medium and high "
                    elif hb>0.33 and hb<0.66 and lb>0.33:
                        significance="is medium "
                    elif hb>0.33 and hb<0.66 and lb>0:
                        significance="is medium and low "

                term+=inp_list[inp]+significance+"AND "
            term=term[:-4]
            if rule_stats[r]["center"]>=0.5:
                term+=") THEN " +action+" (%s)" % rule_stats[r]["center"]
            else:
                term+=") THEN " +action+" (%s)" % rule_stats[r]["center"]
            print(term)
                    
    
    plt.show()

raise

# Action3
Data=pd.read_csv("Data2.csv")
y=Data.Action
x=Data.drop("Action",axis=1)
print(x.shape)
trnData, chkData, trnLbls, chkLbls = train_test_split(x, y, test_size=0.2)

trnData=trnData.to_numpy()
chkData=chkData.to_numpy()
trnLbls=trnLbls.to_numpy()
chkLbls=chkLbls.to_numpy()


# Training
num_epochs = 6000

# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    # Initialize model parameters
    sess.run(fis.init_variables)
    trn_costs = []
    val_costs = []
    time_start = time.time()
    for epoch in range(num_epochs):
        #  Run an update step
        trn_loss, trn_pred = fis.train(sess, trnData, trnLbls)
        # Evaluate on validation set
        val_pred, val_loss = fis.infer(sess, chkData, chkLbls)
        if epoch % 10 == 0:
            print("Train cost after epoch %i: %f" % (epoch, trn_loss))
        if epoch == num_epochs - 1:
            time_end = time.time()
            print("Elapsed time: %f" % (time_end - time_start))
            print("Validation loss: %f" % val_loss)
            # Plot real vs. predicted
            pred = np.vstack((np.expand_dims(trn_pred, 1), np.expand_dims(val_pred, 1)))
            plt.figure(1)
            plt.plot(trn_pred-trnLbls)
            print(trnLbls)
            print(trnLbls.shape)
            #plt.plot(trn_pred)
            print(trn_pred.shape)
        trn_costs.append(trn_loss)
        val_costs.append(val_loss)
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
    fis.plotmfs(sess)
    
    plt.show()

    inp_list=["Seniority ", "Propensity ","Size of Company ", "Contactibility "]
    action="send brochure"
    for r in rule_stats:
        if abs(rule_stats[r]["center"])>0.05:
            #print("rule %s" % r)
            term="\nrule %s: IF(" % r
            for inp in range(4):
                hb= rule_stats[r][inp]["high_bound"]
                lb= rule_stats[r][inp]["low_bound"]
                significance=""
                if lb>=1 or hb<=0: significance="is not related "
                elif hb==1 and lb==0: significance="is in any value "
                else:
                    nothing=0
                    if lb==0:
                        nothing=1
                    extreme=0
                    if hb==1:
                        extreme=1
                    some_little=0
                    if hb<0.34:
                        some_little=1
                    low=0
                    if hb<0.49:
                        low=1
                    some_high=0
                    if lb>0.66:
                        some_high=1
                    high=0
                    if lb>0.51:
                        high=1
                    some_medium=0
                    if lb>0.33 and lb<0.66:
                        some_medium=1
                    if hb>0.33 and hb<0.66:
                        some_medium=1

                    if nothing==1:
                        if low==1:
                            significance="is low "
                        elif some_little==1:
                            significance="is very low "
                        elif hb>0.49:
                            significance="is not extreme high "
                        else:
                            significance="is nothing "
                    elif extreme==1:
                        if high==1:
                            significance="is high "
                        elif some_high==1:
                            significance="is very high "
                        elif lb<0.51:
                            significance="is not extremely low "
                        else:
                            significance="is extremely high "
                    elif low==1 and some_little==1:
                        significance="is low but not zero "
                    elif low==1:
                        significance="is medium low "
                    elif some_little==1:
                        significance="is very low but not zero "
                    elif high==1 and some_high==1:
                        significance="is high but not zero "
                    elif high==1:
                        significance="is medium high "
                    elif some_high==1:
                        significance="is quite high but not extermely high "
                    elif lb>0.33 and lb<0.66 and hb<0.66:
                        significance="is medium "
                    elif lb>0.33 and lb<0.66 and hb<1:
                        significance="is medium and high "
                    elif hb>0.33 and hb<0.66 and lb>0.33:
                        significance="is medium "
                    elif hb>0.33 and hb<0.66 and lb>0:
                        significance="is medium and low "

                term+=inp_list[inp]+significance+"AND "
            term=term[:-4]
            if rule_stats[r]["center"]>=0.5:
                term+=") THEN " +action+" (%s)" % rule_stats[r]["center"]
            else:
                term+=") THEN "+action+" (%s)" % rule_stats[r]["center"]
            print(term)
                        
    
