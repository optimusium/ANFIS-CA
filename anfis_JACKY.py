import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ANFIS:

    def __init__(self, n_inputs, n_rules, n_fuzzy, learning_rate=1e-2):
        self.n = n_inputs
        self.m = n_rules
        self.f = n_fuzzy
        ## self.inputs = tf.placeholder(tf.float32, shape=(None, n_inputs))  # Input
        # Four individual inputs [Seniority, Purchase_Propensity, CompanySize, Contactable]
        fuzzyInput = tf.placeholder(tf.float64)
        self.fuzzyInput0 = self.fuzzyInput1 = self.fuzzyInput2 = self.fuzzyInput3 = self.bias = fuzzyInput
        F = {
            "fuzzyInput0": self.fuzzyInput0,
            "fuzzyInput1": self.fuzzyInput1,
            "fuzzyInput2": self.fuzzyInput2,
            "fuzzyInput3": self.fuzzyInput3
            }
        # Number of linguistic labels for each variable that NN is going to train
        self.fuzzy0 = self.fuzzy1 = self.fuzzy2 = self.fuzzy3 = self.fuzzy4 = self.f  
        self.targets = tf.placeholder(tf.int64, shape=(None))  # Desired output
        
        ## GAUSSIAN MEMBERSHIP FUNCTION
        ## mu = tf.get_variable("mu", [n_rules * n_inputs],
        ##                       initializer=tf.random_normal_initializer(0, 1))  # Means of Gaussian MFS
        ## sigma = tf.get_variable("sigma", [n_rules * n_inputs],
        ##                         initializer=tf.random_normal_initializer(0, 1))  # Standard deviations of Gaussian MFS
        
        ## OUTPUT OF THE CRISP FUNCTION OF CONSEQUENT PARAMETERS
        ## y = tf.get_variable("y", [1, n_rules], initializer=tf.random_normal_initializer(0, 1))  # Sequent centers

        # muF1_junior = tf.get_variable("muF1_junior", [1],
        #                      initializer=tf.random_normal_initializer(0, 1))  # Means of Gaussian MFS
        # sigmaF1_junior = tf.get_variable("sigmaF1_junior", [1],
        #                         initializer=tf.random_normal_initializer(0, 1))  # Standard deviations of Gaussian MFS

        # muF1_senior = tf.get_variable("muF1_senior", [1],
        #                      initializer=tf.random_normal_initializer(0, 1))  # Means of Gaussian MFS
        # sigmaF1_senior = tf.get_variable("sigmaF1_senior", [1],
        #                         initializer=tf.random_normal_initializer(0, 1))  # Standard deviations of Gaussian MFS

        # muF2_noplan = tf.get_variable("muF2_noplan", [1],
        #                      initializer=tf.random_normal_initializer(0, 1))  # Means of Gaussian MFS
        # sigmaF2_noplan = tf.get_variable("sigmaF2_noplan", [1],
        #                         initializer=tf.random_normal_initializer(0, 1))  # Standard deviations of Gaussian MFS

        # muF2_12mths = tf.get_variable("muF2_12mths", [1],
        #                      initializer=tf.random_normal_initializer(0, 1))  # Means of Gaussian MFS
        # sigmaF2_12mths = tf.get_variable("sigmaF2_12mths", [1],
        #                         initializer=tf.random_normal_initializer(0, 1))  # Standard deviations of Gaussian MFS

        # muF3_small = tf.get_variable("muF3_small", [1],
        #                      initializer=tf.random_normal_initializer(0, 1))  # Means of Gaussian MFS
        # sigmaF3_small = tf.get_variable("sigmaF3_small", [1],
        #                         initializer=tf.random_normal_initializer(0, 1))  # Standard deviations of Gaussian MFS

        # muF3_big = tf.get_variable("muF3_big", [1],
        #                      initializer=tf.random_normal_initializer(0, 1))  # Means of Gaussian MFS
        # sigmaF3_big = tf.get_variable("sigmaF3_big", [1],
        #                         initializer=tf.random_normal_initializer(0, 1))  # Standard deviations of Gaussian MFS

        # muF4_nocontact = tf.get_variable("muF4_nocontact", [1],
        #                      initializer=tf.random_normal_initializer(0, 1))  # Means of Gaussian MFS
        # sigmaF4_nocontact = tf.get_variable("sigmaF4_nocontact", [1],
        #                         initializer=tf.random_normal_initializer(0, 1))  # Standard deviations of Gaussian MFS

        # muF4_email = tf.get_variable("muF4_email", [1],
        #                      initializer=tf.random_normal_initializer(0, 1))  # Means of Gaussian MFS
        # sigmaF4_email = tf.get_variable("sigmaF4_email", [1],
        #                         initializer=tf.random_normal_initializer(0, 1))  # Standard deviations of Gaussian MFS



        """
        GAUSSIAN MEMBERSHIP FUNCTION - TRAINABLE PREMISE PARAMETERS
        EACH OF THE 4 INPUT HAS 3 FUZZY LINGUISTIC LABELS & EACH GAUSSIAN MEMBERSHIP FUNCTION HAS 2 PARAMETERS
        THEREFORE, THERE ARE 12 MEANS OF GAUSSIAN MFS & 12 SDs OF GAUSSIAN MFS 
        """
        self.P = {}
        for i in range(self.n):
            for j in range(self.f):
                for k in ["mu", "sigma"]:
                    self.P["{}F{}_{}".format(k, i, j)] = tf.get_variable("{}F{}_{}".format(k, i, j), [1, 1],
                                                            initializer=tf.random_normal_initializer(0, 1),
                                                            dtype=tf.float64)
        
        """
        FIRST-ORDER SUGENO FUZZY MODEL - TRAINABLE CONSEQUENT PARAMETERS
        WITH 4 INPUTS THAT EACH HAS 3 FUZZY LINGUISTIC LABELS WILL GENERATE 81 SETS OF CONSEQUENT PARAMETERS
        EACH SET WILL COMPRISE OF 5 CONSEQUENT PARAMETERS:
            1. coeff_0_* = COEFFICIENTS OF SENIORITY INPUT
            2. coeff_1_* = COEFFICIENTS OF PURCHASE_PROPENSITY INPUT
            3. coeff_2_* = COEFFICIENTS OF COMPANYSIZE INPUT
            4. coeff_3_* = COEFFICIENTS OF CONTACTABLE INPUT
            5. coeff_4_* = CONSTANT
        """
        self.C = {}
        for i in range(self.n + 1):
            for j in range(self.m):
                self.C["coeff_{}_{}".format(i, j)] = tf.get_variable("coeff_{}_{}".format(i, j), [1, 1],
                                                        initializer=tf.random_normal_initializer(0, 1),
                                                        dtype=tf.float64)
            
        """
        GAUSSIAN MEMBERSHIP FUNCTION - TRAINABLE CONSEQUENT PARAMETERS
        THE SINGLE OVERALL OUTPUT HAS 3 FUZZY LINGUISTIC LABELS & 
        EACH GAUSSIAN MEMBERSHIP FUNCTION HAS 2 PARAMETERS
        THEREFORE, THERE ARE 3 MEANS OF GAUSSIAN MFS & 3 SDs OF GAUSSIAN MFS 
        """
        self.E = {}
        for j in range(self.f):
            for k in ["mu", "sigma"]:
                self.E["{}F5_{}".format(k, j)] = tf.get_variable("{}F5_{}".format(k, j), [1, 1],
                                                    initializer=tf.random_normal_initializer(0, 1),
                                                    dtype=tf.float64)

        self.params = tf.trainable_variables()
        
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
                # i.e., 12 tensors of shape=(?, 1) & dtype=float64       
                self.M["self.mem_{}_{}".format(i, j)] = tf.exp(-0.5 * 
                                                        tf.square(
                                                        tf.subtract(F["fuzzyInput{}".format(i)], 
                                                        self.P["muF{}_{}".format(i, j)])) /
                                                        tf.square(self.P["sigmaF{}_{}".format(i, j)]))
        
        """
        ANFIS LAYER 2 - OBTAINING THE FIRING STRENGTH OF RULES
        TAKE THE MINIMUM µ FROM EACH OF THE 81 RULES' SET OF 4 µ 
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
        # self.S is a tensor of shape=(4, ?, 81) & dtype=flost64
        self.S = tf.concat(list(self.s.values()), axis=2)
        # Each input sample will have 81 µ that has the lowest value for their respective rule
        # i.e., self.smallest is a tensor of shape=(?, 81) & dtype=float64
        self.smallest = tf.reduce_min(self.S, axis=0)
   
        """
        ANFIS LAYER 3 - NORMALIZING THE FIRING STRENGTH OF RULES
        NORMALIZE THE 81 µ FOR EACH SAMPLE INPUT BY DIVIDING EACH µ BY THEIR SUMMATION
        """
        # self.summed is a tensor of shape=(?, 1) & dtype=float64
        self.summed = tf.reduce_sum(self.smallest, axis=1, keepdims=True)
        # self.normed is a tensor of shape(?, 81) & dtype=float64
        self.normed = tf.divide(self.smallest, self.summed)
                
        """
        ANFIS LAYER 4 - MULTIPYING NORMED FIRING STRENGTH BY FIRST-ORDER SUGENO FUZZY MODEL
        """
        # self.Z is a tensor of shape(?, 81) & dtype=float64
        self.Z = tf.multiply(self.normed, tf.add_n([        
                                                    tf.scalar_mul(self.C["coeff_0_{}".format(i)], 
                                                                  self.fuzzyInput0),
                                                    tf.scalar_mul(self.C["coeff_1_{}".format(i)], 
                                                                  self.fuzzyInput1),
                                                    tf.scalar_mul(self.C["coeff_2_{}".format(i)], 
                                                                  self.fuzzyInput2),
                                                    tf.scalar_mul(self.C["coeff_3_{}".format(i)], 
                                                                  self.fuzzyInput3),
                                                    tf.scalar_mul(self.C["coeff_4_{}".format(i)], 
                                                                  self.bias)   
                                                    ])
                            )
        
        """
        ANFIS LAYER 5 - COMPUTES THE OVERALL OUTPUT FOR EACH SAMPLE INPUT
        EACH SAMPLE INPUT'S OVERALL OUTPUT IS THE SUMMATION OF ITS 81 OUTPUTS CALCULATED FROM ANFIS LAYER 4 
        """
        # self.Z is a tensor of shape(?, 1) & dtype=float64
        self.O = tf.reduce_sum(self.Z, axis=1, keepdims=True)
            
        """
        ANFIS LAYER 6 - FUZZIFICATION OF OUTPUT VARIABLES
        GAUSSIAN MFS EQUATION - 3 EQUATIONS THAT COVERS ALL 3 LINGUISTIC LABELS
        A. ACTION
            1. DISCARD THIS CONTACT
            2. SEND BROCHURE
            3. CALL AND/OR EMAIL
        """
        self.A = {}
        for i in range(self.f):
            # Each input sample will have 3 µ generated for each of the 3 linguistic labels
            # i.e., 3 tensors of shape=(?, 1) & dtype=float64 
            self.A["self.act_{}".format(i)] = tf.exp(-0.5 * 
                                                        tf.square(
                                                        tf.subtract(self.O, 
                                                        self.E["muF5_{}".format(i)])) /
                                                        tf.square(self.E["sigmaF5_{}".format(i)]))
        
        # Loss function
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                        labels=self.targets, 
                                                        logits=tf.concat(list(self.A.values()), 
                                                        axis=1), 
                                                        name="losses")

        # Optimizer
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        
        # Variable initializer
        self.init_variables = tf.global_variables_initializer()  

    def train(self, sess, trn, targets, bias):
        trn_pred, trn_loss, _ = sess.run([self.A, self.loss, self.optimize], feed_dict={
                                                                             self.fuzzyInput0: trn["trnDataF0"], 
                                                                             self.fuzzyInput1: trn["trnDataF1"],
                                                                             self.fuzzyInput2: trn["trnDataF2"],
                                                                             self.fuzzyInput3: trn["trnDataF3"],
                                                                             self.bias: bias["bias_trn"],
                                                                             self.targets: targets
                                                                             })
        return trn_pred, trn_loss

    def infer(self, sess, chk, targets, bias):
        if targets is None:
            return sess.run(self.A, feed_dict={
                                                self.fuzzyInput0: chk["chkDataF0"], 
                                                self.fuzzyInput1: chk["chkDataF1"],
                                                self.fuzzyInput2: chk["chkDataF2"],
                                                self.fuzzyInput3: chk["chkDataF3"]
                                                 })
        else:
            return sess.run([self.A, self.loss], feed_dict={
                                                             self.fuzzyInput0: chk["chkDataF0"], 
                                                             self.fuzzyInput1: chk["chkDataF1"],
                                                             self.fuzzyInput2: chk["chkDataF2"],
                                                             self.fuzzyInput3: chk["chkDataF3"],
                                                             self.bias: bias["bias_chk"],
                                                             self.targets: targets
                                                             })





    """
    def plotmfs(self, sess):
        rule_stats = {}
        mus = sess.run(self.params[0])
        mus = np.reshape(mus, (self.m, self.n))
        sigmas = sess.run(self.params[1])
        sigmas = np.reshape(sigmas, (self.m, self.n))
        y = sess.run(self.params[2])
        xn = np.linspace(0, 1, 1000)
        for r in range(self.m):
            rule_stats[r+1] = {}
            if r % 4 == 0:
                plt.figure(figsize=(11, 6), dpi=80)
            plt.subplot(2, 2, (r % 4) + 1)
            ax = plt.subplot(2, 2, (r % 4) + 1)
            ax.set_title("Rule %d, sequent center: %f" % ((r + 1), y[0, r]))
            print("\n" + "Sum of y: " + str(np.sum(y)))
            rule_stats[r+1]["center"] = y[0, r] 
            for i in range(self.n):
                rule_stats[r+1][i] = {}
                
                low_bound = mus[r, i] - abs(sigmas[r, i])
                if low_bound < 0: 
                    low_bound = 0
                if low_bound > 1: 
                    low_bound = 1
                
                high_bound = mus[r, i] + abs(sigmas[r, i])
                if high_bound > 1: 
                    high_bound = 1
                if high_bound < 0: 
                    high_bound = 0
                
                rule_stats[r+1][i]["high_bound"] = high_bound
                rule_stats[r+1][i]["low_bound"] = low_bound
                
                plt.plot(xn, np.exp(-0.5 * ((xn - mus[r, i]) ** 2) / (sigmas[r, i] ** 2)) ,"C%s" % i, label="%i" % i)
                ax.legend()
                print("rule %s input %i: mu=%s sigma=%s bound=[%s,%s]" %(r+1,i,mus[r, i],sigmas[r, i], low_bound, high_bound))
                
        return rule_stats, mus, sigmas, y
    """
    






















        
        
        
        # ## GAUSSIAN MFS EQUATION
        # ## self.rul = tf.reduce_prod(
        # ##     tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.inputs, (1, n_rules)), mu)) / tf.square(sigma)),
        # ##                (-1, n_rules, n_inputs)), axis=2)  # Rule activations - Weights from Gaussian MFS
        
        # ## Fuzzy base expansion function:
        # ## num = tf.reduce_sum(tf.multiply(self.rul, y), axis=1) # Summation of all incoming signals 
        # ## den = tf.clip_by_value(tf.reduce_sum(self.rul, axis=1), 1e-12, 1e12) # Summation of firing strength of rules
        
        # ## NORMALIZED FIRING STRENGTHS
        # ## self.wNOR = tf.divide(self.rul, den)
        
        
        
        
        # self.wifi = tf.clip_by_value(tf.reduce_sum(self.rul, axis=1), 1e-12, 1e12)
        
        
        
        # # GAUSSIAN MFS EQUATION - FUZZY1
        # self.rulF1 = tf.reduce_prod(
        #     tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.fuzzy1Input, (1, 1)), muF1)) / tf.square(sigmaF1)),
        #                (-1, 1, self.fuzzy1)), axis=2)  # Rule activations - Weights from Gaussian MFS
        
        # # Fuzzy base expansion function:
        # numF1 = tf.reduce_sum(tf.multiply(self.rulF1, yF1), axis=1) # Summation of all incoming signals 
        # denF1 = tf.clip_by_value(tf.reduce_sum(self.rulF1, axis=1), 1e-12, 1e12) # Summation of firing strength of rules
        
        # self.outF1 = tf.divide(numF1, denF1)
        
        
        
        # # GAUSSIAN MFS EQUATION - FUZZY2
        # self.rulF2 = tf.reduce_prod(
        #     tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.fuzzy2Input, (1, 1)), muF2)) / tf.square(sigmaF2)),
        #                (-1, 1, self.fuzzy2)), axis=2)  # Rule activations - Weights from Gaussian MFS
        
        # # Fuzzy base expansion function:
        # numF2 = tf.reduce_sum(tf.multiply(self.rulF2, yF2), axis=1) # Summation of all incoming signals 
        # denF2 = tf.clip_by_value(tf.reduce_sum(self.rulF2, axis=1), 1e-12, 1e12) # Summation of firing strength of rules
        
        # self.outF2 = tf.divide(numF2, denF2)
        
        
        # # GAUSSIAN MFS EQUATION - FUZZY3
        # self.rulF3 = tf.reduce_prod(
        #     tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.fuzzy3Input, (1, 1)), muF3)) / tf.square(sigmaF3)),
        #                (-1, 1, self.fuzzy3)), axis=2)  # Rule activations - Weights from Gaussian MFS
        
        # # Fuzzy base expansion function:
        # numF3 = tf.reduce_sum(tf.multiply(self.rulF3, yF3), axis=1) # Summation of all incoming signals 
        # denF3 = tf.clip_by_value(tf.reduce_sum(self.rulF3, axis=1), 1e-12, 1e12) # Summation of firing strength of rules
        
        # self.outF3 = tf.divide(numF3, denF3)
        
        
        
        # # GAUSSIAN MFS EQUATION - FUZZY4
        # self.rulF4 = tf.reduce_prod(
        #     tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.fuzzy4Input, (1, 1)), muF4)) / tf.square(sigmaF4)),
        #                (-1, 1, self.fuzzy4)), axis=2)  # Rule activations - Weights from Gaussian MFS
        
        # # Fuzzy base expansion function:
        # numF4 = tf.reduce_sum(tf.multiply(self.rulF4, yF4), axis=1) # Summation of all incoming signals 
        # denF4 = tf.clip_by_value(tf.reduce_sum(self.rulF4, axis=1), 1e-12, 1e12) # Summation of firing strength of rules
        
        # self.outF4 = tf.divide(numF4, denF4)
        
        
        
        
        
        
        # # self.out = tf.divide(num, den) # Overall output after weight normalization
        
        # # if tf.nn.softmax(tf.divide(num, den)) < 0.5:
        # #     self.out = 0
        # # elif tf.nn.softmax(tf.divide(num, den)) >= 0.5:
        # #     self.out = 1
        
        # # self.out = tf.map_fn(lambda x: x * x, elems) 
        
        # # self.out = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=tf.divide(num, den)))
                
        # # self.out = tf.cond(tf.sigmoid(tf.divide(num, den)) >= tf.constant(0.5), 1, 0)
        
        
        # # x = tf.sigmoid(tf.divide(num, den))
        # # y = tf.constant(0.5)
        # # self.out = tf.math.greater_equal(x, y)
        
        # self.out = tf.tanh(tf.divide(num, den))
        
        # self.sigmoid = tf.sigmoid(tf.divide(num, den))
        

        
        # self.loss = tf.losses.hinge_loss(self.targets, self.out)
        
        
        # # self.loss = tf.nn.sigmoid_cross_entropy_with_logits(self.targets, self.out)

        # # self.loss = tf.losses.huber_loss(self.targets, self.out)  # Loss function computation
        # # self.loss = tf.losses.absolute_difference(self.targets, self.out)
        # # Other loss functions for regression, uncomment to try them:
        # # loss = tf.sqrt(tf.losses.mean_squared_error(target, out))
        # # loss = tf.losses.absolute_difference(target, out)
        # self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)  # Optimization step
        # # Other optimizers, uncomment to try them:
        # #self.optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss)
        # # self.optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)
        # self.init_variables = tf.global_variables_initializer()  # Variable initializer





    # def infer(self, sess, x, targets=None):
    #     if targets is None:
    #         return sess.run(self.out, feed_dict={self.inputs: x})
    #     else:
    #         return sess.run([self.out, self.loss], feed_dict={self.inputs: x, self.targets: targets})

    # def train(self, sess, x, targets):
    #     yp, l, _, sig, wifi = sess.run([self.out, self.loss, self.optimize, self.sigmoid, self.wifi], feed_dict={self.inputs: x, self.targets: targets})
    #     return l, yp, sig, wifi

    # # def firstEpoch(self, sess, x, tf_gradnorm_summary, targets):
    # #     yp, l, _, gradnorm = sess.run([self.out, self.loss, self.optimize, tf_gradnorm_summary], feed_dict={self.inputs: x, self.targets: targets})
    # #     return l, yp, gradnorm 

    # def plotmfs(self, sess):
    #     rule_stats = {}
    #     mus = sess.run(self.params[0])
    #     mus = np.reshape(mus, (self.m, self.n))
    #     sigmas = sess.run(self.params[1])
    #     sigmas = np.reshape(sigmas, (self.m, self.n))
    #     y = sess.run(self.params[2])
    #     xn = np.linspace(0, 1, 1000)
    #     for r in range(self.m):
    #         rule_stats[r+1] = {}
    #         if r % 4 == 0:
    #             plt.figure(figsize=(11, 6), dpi=80)
    #         plt.subplot(2, 2, (r % 4) + 1)
    #         ax = plt.subplot(2, 2, (r % 4) + 1)
    #         ax.set_title("Rule %d, sequent center: %f" % ((r + 1), y[0, r]))
    #         print("\n" + "Sum of y: " + str(np.sum(y)))
    #         rule_stats[r+1]["center"] = y[0, r] 
    #         for i in range(self.n):
    #             rule_stats[r+1][i] = {}
                
    #             low_bound = mus[r, i] - abs(sigmas[r, i])
    #             if low_bound < 0: 
    #                 low_bound = 0
    #             if low_bound > 1: 
    #                 low_bound = 1
                
    #             high_bound = mus[r, i] + abs(sigmas[r, i])
    #             if high_bound > 1: 
    #                 high_bound = 1
    #             if high_bound < 0: 
    #                 high_bound = 0
                
    #             rule_stats[r+1][i]["high_bound"] = high_bound
    #             rule_stats[r+1][i]["low_bound"] = low_bound
                
    #             plt.plot(xn, np.exp(-0.5 * ((xn - mus[r, i]) ** 2) / (sigmas[r, i] ** 2)) ,"C%s" % i, label="%i" % i)
    #             ax.legend()
    #             print("rule %s input %i: mu=%s sigma=%s bound=[%s,%s]" %(r+1,i,mus[r, i],sigmas[r, i], low_bound, high_bound))
                
    #     return rule_stats, mus, sigmas, y
    
    
    
    
    
    
    
    
    
    
    