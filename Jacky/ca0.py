import os
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from anfis import ANFIS

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
                      chkLbls=None, bias=None, n_inputs=None, n_fuzzy=None,
                      epochs=500):
    num_epochs = epochs
    # Initialize session to make computations on the Tensorflow graph
    with tf.Session() as sess:
        
        """
        Visualization of the model in TensorBoard
        Type this in command prompt 'tensorboard --logdir=summaries',
        then go to 'http://localhost:6006/' to view the model
        """
        if not os.path.exists('summaries'):
            os.mkdir('summaries')
        if not os.path.exists(os.path.join('summaries','first')):
            os.mkdir(os.path.join('summaries','first'))
        summ_writer = tf.summary.FileWriter(os.path.join('summaries','first'), 
                                            sess.graph)
        
        # Initialize model parameters & some lists
        sess.run(fis.init_variables)
        trn_costs = []
        val_costs = []
        trn_predList = []
        val_predList = []
        
        # Training & Validation
        time_start = time.time()
        for epoch in range(num_epochs):
            trn_pred, trn_loss, trn_P, trn_C, trn_E, trn_sugeno = fis.train(sess, trn, trnLbls, bias)
            val_pred, val_loss, val_P, val_C, val_E, val_sugeno = fis.infer(sess, chk, chkLbls, bias)
            if epoch % 100 == 0:
                print("Training loss after epoch %i: %f" % (epoch, trn_loss))
            if epoch == num_epochs - 1:
                time_end = time.time()
                print("\n" + "Elapsed time: %f" % (time_end - time_start))
                print("Validation loss: %f" % val_loss)
            trn_costs.append(trn_loss)
            val_costs.append(val_loss)
            trn_predList.append(trn_pred)
            val_predList.append(val_pred)
            
            # Plot the accuracy
            plt.figure(1)
            plt.tight_layout(pad=1.0)
            plt.subplot(2, 1, 1)
            plt.plot(trn_predList)
            plt.title("Training accuracy")
            plt.ylabel("Accuracy(%)")
            plt.xlabel("Epochs")
            plt.subplot(2, 1, 2)
            plt.plot(val_predList)
            plt.title("Validation accuracy")
            plt.ylabel("Accuracy(%)")
            plt.xlabel("Epochs")

        # Plot the losses over epochs
        plt.figure(2)
        plt.tight_layout()
        plt.subplots_adjust(hspace = 0.2)
        plt.subplot(2, 1, 1)
        plt.plot(trn_costs)
        plt.title("Training loss, Learning rate =" + str(alpha), fontsize=10)
        plt.ylabel('Losses')
        plt.xlabel('Epochs')
        plt.subplot(2, 1, 2)
        plt.plot(val_costs)
        plt.title("Validation loss, Learning rate =" + str(alpha), fontsize=10)
        plt.ylabel('Losses')
        plt.xlabel('Epochs')
        
        # Plot the Gaussian MFS of all inputs' linguistic labels
        fis.plotmfs_Premise(trn_P, n_inputs, n_fuzzy)
        
        # Plot the Gaussian MFS of actions' linguistic labels
        fis.plotmfs_Consequent(trn_E, n_fuzzy)
        
        # Print the Premise Parameters of all inputs' linguistic labels
        premiseParams = pd.DataFrame(np.reshape(list(trn_P.values()), (12, 2)), 
                                     columns=['mu', 'sigma'],
                                     index=[
                                             "Junior", "Mid", "Senior",
                                             "No_plan", "3_yrs", "12_mths",
                                             "Small", "Medium", "Big",
                                             "Don't_contact", "Only_content", "Can email/call"                                   
                                            ])
        print("\n" + "Premise Parameters of All 12 Inputs' Linguistic Lables" + "\n", premiseParams)
        
        # Print the Consequent Parameters of actions' linguistic labels
        conseqParams = pd.DataFrame(np.reshape(list(trn_E.values()), (3, 2)),
                                        columns=['mu', 'sigma'],
                                        index=["Discard", "Brochure", "Email/Call"])
        print("\n" + "Consequent Parameters of All 3 Actions' Linguistic Labels" + "\n", conseqParams)
        
        # Print the first-order Sugeno model's coefficients for all 81 rules
        coeffParams = pd.DataFrame(np.reshape(list(trn_C.values()), (81, 5)),
                                        columns=["Seniority", "Purchase_Propensity",
                                                 "Company_Size", "Contactable", "Bias"])
        print("\n" + "Coefficients of All 81 Rules" + "\n", coeffParams)
            
    return trn_pred, trn_loss, trn_P, trn_C, trn_E, trn_sugeno, val_pred, val_loss, val_P, val_C, val_E, val_sugeno

# ANFIS params and Tensorflow graph initialization
n = 4  # number of inputs
f = 3 # number of linguistic labels per input variable
m = f ** n  # number of rules = 81
alpha = 0.0009  # learning rate
tf.reset_default_graph()
fis = ANFIS(n_inputs=n, n_rules=m, n_fuzzy=f, learning_rate=alpha)

# Print model summary
model_summary()

# Get data
trn, chk, trnLbls, chkLbls, bias = train_testData("Data_combined.csv", n)

# Train & plot the ANFIS model
trn_pred, trn_loss, trn_P, trn_C, trn_E, trn_sugeno, val_pred, val_loss, val_P, val_C, val_E, val_sugeno = trainingNplotting(fis, trn, trnLbls, chk, chkLbls, bias, n, f)

