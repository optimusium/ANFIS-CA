import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ANFIS:

    def __init__(self, n_inputs, n_rules, n_fuzzy, learning_rate=1e-2):
        self.n = n_inputs
        self.m = n_rules
        self.f = n_fuzzy
        """ Four individual inputs [Seniority, Purchase_Propensity, Company_Size, Contactable] """
        self.fuzzyInput0 = tf.placeholder(tf.float32, shape=(None, 1))
        self.fuzzyInput1 = tf.placeholder(tf.float32, shape=(None, 1))
        self.fuzzyInput2 = tf.placeholder(tf.float32, shape=(None, 1))
        self.fuzzyInput3 = tf.placeholder(tf.float32, shape=(None, 1))
        self.bias = tf.placeholder(tf.float32, shape=(None, 1)) # Bias terms for first-order Sugeno fuzzy model
        F = {
            "fuzzyInput0": self.fuzzyInput0,
            "fuzzyInput1": self.fuzzyInput1,
            "fuzzyInput2": self.fuzzyInput2,
            "fuzzyInput3": self.fuzzyInput3
            }
        # Number of linguistic labels for each variable that NN is going to train
        self.fuzzy0 = self.fuzzy1 = self.fuzzy2 = self.fuzzy3 = self.fuzzy4 = self.f  
        """
        Desired output: 
        0 represented Action = Discard
        0.5 represented Action = Brochure
        1 represented Action = Email/Call
        """
        self.targets = tf.placeholder(tf.int32, shape=(None, 1))  
        
        """INITIALIZING TRAINABLE PARAMETERS""" 
        """
        GAUSSIAN MEMBERSHIP FUNCTION - TRAINABLE PREMISE PARAMETERS
        EACH OF THE 4 INPUT HAS 3 FUZZY LINGUISTIC LABELS & EACH GAUSSIAN MEMBERSHIP FUNCTION HAS 2 PARAMETERS
        THEREFORE, THERE ARE 12 MEANS OF GAUSSIAN MFS & 12 SDs OF GAUSSIAN MFS 
        """
        self.P = {}
        for i in range(self.n):
            for j in range(self.f):
                for k in ["mu", "sigma"]:
                    if k == "mu":
                        init_mean=0.1
                        if j==1: init_mean=0.5
                        if j==2: init_mean=0.9
                        self.P["{}F{}_{}".format(k, i, j)] = tf.get_variable("{}F{}_{}".format(k, i, j), [1],
                                                                initializer=tf.random_uniform_initializer(init_mean-0.1, init_mean+0.1), constraint=lambda t: tf.clip_by_value(t, 0.001, 0.999),
                                                                dtype=tf.float32, regularizer = tf.contrib.layers.l2_regularizer(scale=0.1))
                    else:
                        self.P["{}F{}_{}".format(k, i, j)] = tf.get_variable("{}F{}_{}".format(k, i, j), [1],
                                                                initializer=tf.random_uniform_initializer(0, 1), constraint=lambda t: tf.clip_by_value(t, 0.0, 0.33), 
                                                                dtype=tf.float32, regularizer = tf.contrib.layers.l2_regularizer(scale=0.1))

        """
        FIRST-ORDER SUGENO FUZZY MODEL - TRAINABLE CONSEQUENT PARAMETERS
        WITH 4 INPUTS THAT EACH HAS 3 FUZZY LINGUISTIC LABELS WILL GENERATE 81 SETS OF CONSEQUENT PARAMETERS
        EACH SET WILL COMPRISE OF 5 CONSEQUENT PARAMETERS:
            1. coeff_0_* = COEFFICIENTS OF SENIORITY INPUT
            2. coeff_1_* = COEFFICIENTS OF PURCHASE_PROPENSITY INPUT
            3. coeff_2_* = COEFFICIENTS OF COMPANYSIZE INPUT
            4. coeff_3_* = COEFFICIENTS OF CONTACTABLE INPUT
            5. coeff_4_* = CONSTANT/BIAS
        """
        self.C = {}
        for i in range(self.n + 1):
            for j in range(self.m):
                self.C["coeff_{}_{}".format(i, j)] = tf.get_variable("coeff_{}_{}".format(i, j), [1],
                                                        initializer=tf.random_uniform_initializer(0, 1),
                                                        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1),
                                                        dtype=tf.float32)
            
        """
        GAUSSIAN MEMBERSHIP FUNCTION - TRAINABLE CONSEQUENT PARAMETERS
        THE SINGLE OVERALL OUTPUT OF THE ANFIS MODEL HAVE 3 FUZZY LINGUISTIC LABELS & 
        EACH GAUSSIAN MEMBERSHIP FUNCTION HAS 2 PARAMETERS
        THEREFORE, THERE ARE 3 MEANS OF GAUSSIAN MFS & 3 SDs OF GAUSSIAN MFS 
        """
        self.E = {}
        for j in range(3):
            for k in ["mu", "sigma"]:
                if k == "mu":
                        init_mean=0.1
                        if j==1: init_mean=0.5
                        if j==2: init_mean=0.9
                        self.E["{}F5_{}".format(k, j)] = tf.get_variable("{}F5_{}".format(k, j), [1],
                                                                initializer=tf.random_uniform_initializer(init_mean-0.1, init_mean+0.1), constraint=lambda t: tf.clip_by_value(t, 0.001, 0.999),
                                                                dtype=tf.float32, regularizer = tf.contrib.layers.l2_regularizer(scale=0.1))
                else:
                        self.E["{}F5_{}".format(k, j)] = tf.get_variable("{}F5_{}".format(k, j), [1],
                                                                initializer=tf.random_uniform_initializer(0, 1), constraint=lambda t: tf.clip_by_value(t, 0.0, 0.33), 
                                                                dtype=tf.float32, regularizer = tf.contrib.layers.l2_regularizer(scale=0.1))

        self.params = tf.trainable_variables()
        
        """ANFIS MODEL WITH MAMDANI'S FUZZY LINGUISTIC LABELS OUTPUT"""
        """
        ANFIS LAYER 1 - FUZZIFICATION OF INPUT VARIABLES
        GAUSSIAN MFS EQUATION - 12 EQUATIONS THAT COVERS ALL 12 LINGUISTIC LABELS
        A. SENIORITY
            1. JUNIOR
            2. MID
            3. SENIOR
        B. PURCHASE_PROPENSITY
            4. NO PLAN
            5. WITHIN THE NEXT 3 YEARS
            6. WITHIN THE NEXT 12 MONTHS
        C. COMPANYSIZE
            7. SMALL
            8. MEDIUM
            9. BIG
        D. CONTACTABLE
            10. DON'T CONTACT ME
            11. SEND ONLY CONTENT TO ME
            12. EMAIL OR CALL ME
        """
        self.M = {}
        for i in range(self.n):
            for j in range(self.f):
                # Each input sample will have 12 µ generated for each of the 12 linguistic labels    
                self.M["self.mem_{}_{}".format(i, j)] = tf.nn.relu(tf.exp(-0.5 * 
                                                        tf.square(
                                                        tf.subtract(F["fuzzyInput{}".format(i)], 
                                                        self.P["muF{}_{}".format(i, j)])) /
                                                        tf.square(self.P["sigmaF{}_{}".format(i, j)])))
                                
        """
        ANFIS LAYER 2 - OBTAINING THE FIRING STRENGTH OF RULES
        TAKE THE PRODUCT FROM EACH OF THE 81 RULES' SET OF 4 µ 
        """
        self.s = {}
        z = 0
        for i in range(self.fuzzy0):
            for j in range(self.fuzzy1):
                for k in range(self.fuzzy2):
                    for l in range(self.fuzzy3):
                        self.s["firingStrength_{}".format(z)] = [
                                                                self.M["self.mem_0_{}".format(i)],
                                                                self.M["self.mem_1_{}".format(j)],
                                                                self.M["self.mem_2_{}".format(k)],
                                                                self.M["self.mem_3_{}".format(l)],
                                                                ]
                        z += 1
        self.S = tf.concat(list(self.s.values()), axis=2)
        # Each input sample will have the product of 4 µ for their respective rule
        self.prod = tf.reduce_prod(self.S, axis=0)
   
        """
        ANFIS LAYER 3 - NORMALIZING THE FIRING STRENGTH OF RULES
        NORMALIZE THE 81 µ FOR EACH SAMPLE INPUT BY DIVIDING EACH µ BY THEIR SUMMATION
        """
        self.summed = tf.reduce_sum(self.prod, axis=1, keepdims=True)
        self.normed = tf.divide(self.prod, self.summed)
                
        """
        ANFIS LAYER 4 - MULTIPYING NORMED FIRING STRENGTH BY FIRST-ORDER SUGENO FUZZY MODEL
        """        
        self.Z = tf.nn.relu(tf.multiply(self.normed, tf.add_n([        
                                                    tf.scalar_mul(tf.squeeze(self.C["coeff_0_{}".format(i)]), 
                                                                  self.fuzzyInput0),
                                                    tf.scalar_mul(tf.squeeze(self.C["coeff_1_{}".format(i)]), 
                                                                  self.fuzzyInput1),
                                                    tf.scalar_mul(tf.squeeze(self.C["coeff_2_{}".format(i)]), 
                                                                  self.fuzzyInput2),
                                                    tf.scalar_mul(tf.squeeze(self.C["coeff_3_{}".format(i)]), 
                                                                  self.fuzzyInput3),
                                                    tf.scalar_mul(tf.squeeze(self.C["coeff_4_{}".format(i)]), 
                                                                  self.bias)   
                                                    ])
                            ))
        
        """
        ANFIS LAYER 5 - COMPUTES THE OVERALL OUTPUT FOR EACH SAMPLE INPUT (i.e., OUTPUT OF THE SUGENO MODEL)
        EACH SAMPLE INPUT'S OVERALL OUTPUT IS THE SUMMATION OF ITS 81 OUTPUTS CALCULATED FROM ANFIS LAYER 4 
        """
        self.SUGENO =tf.reduce_sum(self.Z, axis=1, keepdims=True)
            
        """
        ANFIS LAYER 6 - FUZZIFICATION OF OUTPUT VARIABLES (i.e., EXTENDING ANFIS TO THE MAMDANI MODEL)
        GAUSSIAN MFS EQUATION - 3 EQUATIONS THAT COVERS ALL 3 LINGUISTIC LABELS
        A. ACTION
            1. DISCARD THIS CONTACT
            2. SEND BROCHURE
            3. CALL AND/OR EMAIL
        """
        self.MAMDANI = {}
        for i in range(self.f):
            # Each input sample will have 3 µ generated for each of the 3 linguistic labels
            self.MAMDANI["self.act_{}".format(i)] = tf.nn.relu(tf.exp(-0.5 * 
                                                        tf.square(
                                                        tf.subtract(self.SUGENO, 
                                                        self.E["muF5_{}".format(i)])) /
                                                        tf.square(self.E["sigmaF5_{}".format(i)])))
            
        self.loss = tf.compat.v1.losses.huber_loss(
            labels=tf.squeeze(self.targets), predictions=self.SUGENO)
        
        # Optimizer
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # Variable initializer
        self.init_variables = tf.global_variables_initializer()  

    def train(self, sess, trn, targets, bias):
        trn_pred, trn_loss, _, trn_P, trn_C, trn_E = sess.run([self.SUGENO, self.loss, 
                                                               self.optimize, self.P, 
                                                               self.C, self.E], 
                                         feed_dict={
                                                   self.fuzzyInput0: 
                                                   np.reshape(trn["trnDataF0"], (len(trn["trnDataF0"]), 1)),
                                                   self.fuzzyInput1:
                                                   np.reshape(trn["trnDataF1"], (len(trn["trnDataF1"]), 1)),
                                                   self.fuzzyInput2:
                                                   np.reshape(trn["trnDataF2"], (len(trn["trnDataF2"]), 1)),
                                                   self.fuzzyInput3:
                                                   np.reshape(trn["trnDataF3"], (len(trn["trnDataF3"]), 1)),
                                                   self.bias: 
                                                   np.reshape(bias["bias_trn"], (len(bias["bias_trn"]), 1)),    
                                                   self.targets: 
                                                   np.reshape(targets, (len(targets), 1))
                                                   })
                                
        return trn_pred, trn_loss, trn_P, trn_C, trn_E

    def infer(self, sess, chk, targets, bias):
        if targets is None:
            return sess.run(self.MAMDANI, feed_dict={
                                                self.fuzzyInput0: chk["chkDataF0"], 
                                                self.fuzzyInput1: chk["chkDataF1"],
                                                self.fuzzyInput2: chk["chkDataF2"],
                                                self.fuzzyInput3: chk["chkDataF3"]
                                                 })
        else:
            chk_pred, chk_loss, chk_P, chk_C, chk_E = sess.run([self.SUGENO, self.loss, 
                                                                self.P, self.C, self.E], 
                                                feed_dict={
                                                   self.fuzzyInput0: 
                                                   np.reshape(chk["chkDataF0"], (len(chk["chkDataF0"]), 1)),
                                                   self.fuzzyInput1:
                                                   np.reshape(chk["chkDataF1"], (len(chk["chkDataF1"]), 1)),
                                                   self.fuzzyInput2:
                                                   np.reshape(chk["chkDataF2"], (len(chk["chkDataF2"]), 1)),
                                                   self.fuzzyInput3:
                                                   np.reshape(chk["chkDataF3"], (len(chk["chkDataF3"]), 1)),
                                                   self.bias: 
                                                   np.reshape(bias["bias_chk"], (len(bias["bias_chk"]), 1)),    
                                                   self.targets: 
                                                   np.reshape(targets, (len(targets), 1))
                                                            })
            
            return chk_pred, chk_loss, chk_P, chk_C, chk_E
                
    def plotmfs_Premise(self, trn_P, n_inputs, n_fuzzy):
        inputName = ["Seniority", "Purchase_Propensity", "Company_Size", "Contactable"]
        fuzzyName =[
                       ["Junior", "Mid", "Senior"],
                       ["No_plan", "3_yrs", "12_mths"],
                       ["Small", "Medium", "Big"],
                       ["Don't_contact", "Only_content", "Can email/call"]
                   ]
        x = np.arange(0, 1.01, 0.01).tolist()
        for i in range(n_inputs):
            plt.figure(5 + i)
            for j in range(n_fuzzy):
                y = []
                for k in range(len(x)):
                    y.append(np.exp(-0.5 * np.square(x[k] - trn_P["muF{}_{}".format(i, j)]) / 
                                                  np.square(trn_P["sigmaF{}_{}".format(i, j)])))
                plt.plot(x, y, "C{}".format(j), label = "{}".format(fuzzyName[i][j]))
                plt.title("{} - Membership Function Curves".format(inputName[i]))
                plt.xlabel("{}".format(inputName[i]))
                plt.legend(loc=4, prop={'size': 7.5})
                
    def plotmfs_Consequent(self, trn_E, n_fuzzy):
        fuzzyName = ["Discard", "Brochure", "Email/Call"]
        x = np.arange(0, 1.01, 0.01).tolist()
        plt.figure(9)
        for i in range(n_fuzzy):
            y = []
            for j in range(len(x)):
                    y.append(np.exp(-0.5 * np.square(x[j] - trn_E["muF5_{}".format(i)]) / 
                                                  np.square(trn_E["sigmaF5_{}".format(i)])))
            plt.plot(x, y, "C{}".format(i), label = "{}".format(fuzzyName[i]))
        plt.title("Actions - Membership Function Curves")
        plt.xlabel("Actions")
        plt.legend(loc=4, prop={'size': 7.5})

