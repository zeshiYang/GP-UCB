import warnings
import numpy as np
from scipy.optimize import minimize
from IPython import embed
from surrogatemodels import GP,MultiOutputGP
from sklearn.cluster import KMeans
import scipydirect
'''
some useful utils functions needed by BO
'''


def generateInitPoints(bounds, numInitPoints):
    '''
    use kmeans to generate intial evaluation points needed by Bayesian Optimization
    reasons: zeshi's thoughts:
            1.since we have limited number of points for initization
              if we use np.random, the generated points may occurs in a small region 
            2.Instead we generate many points and use kmeans to find center of clusters,
              in this way the generated points are more "uniform"
    inputs : bounds: numpy array (2, d)
             numInitPoints : int
    outputs : numpy array (numInitPoints, d)
    '''
    dim = bounds.shape[1]
    interval = (bounds[1,:] - bounds[0, :]).reshape((1, -1))
    if(numInitPoints == 1):
        initPoints = (bounds[0, :] + bounds[1, :])/2
        initPoints = initPoints.reshape((1,-1))
    else:
        if(numInitPoints < 10):
            numRandPoints = 200
        else:
            numRandPoints = 10 * numInitPoints
        initPointsCandidates = np.random.rand(numRandPoints, dim) * interval + bounds[0,:].reshape((1, -1))
        kmeans = KMeans(n_clusters = numInitPoints, random_state=0).fit(initPointsCandidates)
        initPoints = kmeans.cluster_centers_

    return initPoints


class BO(object):
    '''
    class of standard GP-UCB
    '''

    def __init__(self, functions, paras):
        '''
        functions: objective functions 
        params: include paras_opt and paras_gp for optimization and gaussian process respectively
        '''
        self.paras_gp = paras["gp"]
        self.paras_opt = paras["opt"]
        self.iterations = 0
        self.num_iterations = self.paras_opt["num_iterations"]
        self.obj_function = functions

        self.bounds = self.paras_gp["bounds"] #input boundary
        self.dim_inputs = self.bounds.shape[1] 
        self.bounds_lower = self.bounds[0]
        self.bounds_upper = self.bounds[1]
        self.gp = GP(self.paras_gp) # surrogate model
     
        #track information to plot learning curves
        self.budget_track = []
        self.yMax_track=[]
        self.budget_current = 0 
        self.yMax = -1e6

    def initializeGPs(self):
        # initialize gaussian process
        numInitPoints = self.paras_opt["num_init_points"]
        x = generateInitPoints(self.bounds, numInitPoints)
        print("inital points:{}".format(x))
        y = self.evaluate(x)
        self.gp.addData(x, y)
        self.gp.optimizeModel()

    def optimize(self):
        # main loop of GP-UCB
        print("start optimization")
        if (self.iterations == 0):
            self.initializeGPs()
        while (self.iterations < self.num_iterations):
            print("X:{}".format(np.concatenate(self.gp.X), 0))
            print("Y:{}".format(np.concatenate(self.gp.Y), 0))
            self.gp.optimizeModel()
            self.iterations += 1
            x = self.optimize_acq_f()
            acq, mu, sigma, beta = self.acq_f_info(x)
            print("#########selected sample informations#############")
            print("aquisitoin values:{}".format(acq))
            print("mean value:{}".format(mu))
            print("variance value:{}".format(sigma))
            print("beta value:{}".format(beta))
            print("x:{}".format(x))
            print("##########################################")
            y = self.evaluate(x)
            print("y:{}".format(y))
            self.gp.addData(x, y)
            self.record(x, y)
            #self.plot()
        print("budget track:{}".format(self.budget_track))
        print("ymax track:{}".format(self.yMax_track))


    def evaluate(self, x):
        # evaluate function
        y = self.obj_function(x)
        y = y.reshape((-1, 1))
        return y
    
    # def plot(self):
    #     import matplotlib.pyplot as plt
    #     from IPython import embed
    #     fig, ax = plt.subplots(figsize=(7,5.5))

    #     x = np.arange(100)*0.1
    #     x = x.reshape((-1,self.dim_inputs))
    #     embed()
    #     y = self.evaluate(x)
    #     acqs,means,sigmas,_ = self.acq_f_info(x)
    #     ax.plot(x, y, linestyle=':', color = 'tab:green', linewidth = 4, label = "y = x*sin(x)")
    #     ax.plot(x, means, color = 'tab:red', linewidth = 2, label = "Prediction")
    #     #ax.plot(x, acqs - 4, color = "tab:blue", linewidth = 2, label = "acquisition function")
    #     #ax.fill_between(x.reshape((-1)), np.array([-4]*300), acqs.reshape((-1))-4, color='tab:blue',alpha=0.2)
    #     ax.fill_between(x.reshape((-1)), (means - 2* sigmas).reshape((-1)), (means + 2*sigmas).reshape((-1)), color='tab:red',alpha=0.2, label = "0.95 confidence interval")
    #     x = np.concatenate(self.gp.X, 0)[:-1,0]
    #     y = np.concatenate(self.gp.Y, 0)[:-1,0]
    #     ax.scatter(x, y, marker = 'o', s=[80]*x.shape[0], c = 'tab:red', label = 'Observations')
    #     plt.legend(loc = "best", fontsize = 20)
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig("./" + str(self.iterations)+".png")

    def optimize_acq_f(self, n_iter=50, method = "DIRECT"):
                # optimization of aquisition function to get next query point x
        def obj_LBFGS(x):
            return -self.acq_f(x)

        x_tries = np.random.uniform(self.bounds[0, :], self.bounds[1, :],
                                    size=(10000, self.bounds.shape[1]))
        x_seeds = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=(n_iter, self.bounds.shape[1]))
        ys = -obj_LBFGS(x_tries)
        x_max = x_tries[ys.argmax()].reshape((1, -1))
        max_acq = ys.max()
        if(method == "LBFGS"):
            for x_try in x_seeds:
                # Find the minimum of minus the acquisition function
                res = minimize(obj_LBFGS,
                                x_try.reshape(1, -1),
                                bounds=self.reformat_bounds(self.bounds),
                                method="L-BFGS-B")

                # See if success
                if not res.success:
                    continue

                # Store it if better than previous minimum(maximum).
                if max_acq is None or -res.fun[0] > max_acq:
                    x_max = res.x
                    max_acq = -res.fun[0]
        elif(method == "DIRECT"):
            ys = -obj_LBFGS(x_tries)
            x_max = x_tries[ys.argmax()].reshape((1, -1))
            max_acq = ys.max()
            x = scipydirect.minimize(obj_LBFGS, self.reformat_bounds(self.bounds)).x
            acq = -obj_LBFGS(x)[0,0]
            if (acq > max_acq):
                x_max = x
        else:
            raise NotImplementedError

        return np.clip(x_max, self.bounds[0, :], self.bounds[1, :]).reshape((1, -1))

    def acq_f(self, x):
        # aquisition funcition in GP-UCB
        x = np.reshape(x, (-1, self.paras_gp["input_dim"]))
        means, variances = self.gp.predict(x)
        sigmas = np.sqrt(variances)
        means = means.reshape((-1, 1))
        sigmas = sigmas.reshape((-1, 1))
        beta = np.sqrt(self.paras_gp["input_dim"] * np.log(2 * self.iterations * 1 + 1))
        acq = means + beta * sigmas 
        return acq

    def acq_f_info(self, x):
        #print informations in acquisition funtions: acquistion values, means, sigmas, beta
        x = np.reshape(x, (-1, self.paras_gp["input_dim"]))
        means, variances = self.gp.predict(x)
        sigmas = np.sqrt(variances)
        means = means.reshape((-1, 1))
        sigmas = sigmas.reshape((-1, 1))
        beta = np.sqrt(self.paras_gp["input_dim"] * np.log(2 * self.iterations * 1 + 1))
        acq = means + beta * sigmas 
        return acq, means, sigmas, beta

    def record(self, x, y):
        # record information to plot learning curve
        if(y > self.yMax):
            self.yMax = y
            self.xMax = x       
        self.yMax_track.append(self.yMax)
        print("iteration:{}".format(self.iterations))
        print("yMax:{}".format(self.yMax))
        
    def reformat_bounds(self, bounds):
        assert len(bounds) == 2, "unexpected number of bounds"
        return list(zip(*bounds))  



class MaximumEntropySearch(object):
    '''
    class of Maximum Entropy Search 
    '''

    def __init__(self, functions, paras):
        '''
        functions: objective functions 
        params: include paras_opt and paras_gp for optimization and gaussian process respectively
        '''
        self.paras_gp = paras["gp"]
        self.paras_opt = paras["opt"]
        self.iterations = 0
        self.num_iterations = self.paras_opt["num_iterations"]
        self.obj_function = functions

        self.bounds = self.paras_gp["bounds"] #input boundary
        self.dim_inputs = self.bounds.shape[1]
        self.dim_outputs = self.paras_gp["output_dim"] 
        self.bounds_lower = self.bounds[0]
        self.bounds_upper = self.bounds[1]
        self.gp = MultiOutputGP(self.paras_gp) # surrogate model
     
        #track information to plot learning curves
        self.budget_track = []
        self.budget_current = 0 

    def initializeGPs(self):
        # initialize gaussian process
        numInitPoints = self.paras_opt["num_init_points"]
        x = generateInitPoints(self.bounds, numInitPoints)
        print("inital points:{}".format(x))
        y = self.evaluate(x)
        self.gp.addData(x, y)
        self.gp.optimizeModel()

    def optimize(self):
        # main loop of MES
        print("start optimization")
        if (self.iterations == 0):
            self.initializeGPs()
        while (self.iterations < self.num_iterations):
            #self.plot()
            print("X:{}".format(np.concatenate(self.gp.X), 0))
            print("Y:{}".format(np.concatenate(self.gp.Y), 0))
            self.gp.optimizeModel()
            self.iterations += 1
            x = self.optimize_acq_f()
            acq, mu, sigma = self.acq_f_info(x)
            print("#########selected sample informations#############")
            print("aquisitoin values:{}".format(acq))
            print("mean value:{}".format(mu))
            print("variance value:{}".format(sigma))
            print("x:{}".format(x))
            print("##########################################")
            y = self.evaluate(x)
            print("y:{}".format(y))
            self.gp.addData(x, y)
            self.record(x, y)
        print("budget track:{}".format(self.budget_track))

    def evaluate(self, x):
        # evaluate function
        y = self.obj_function(x)
        y = y.reshape((-1, self.dim_outputs))
        return y
    
    def plot(self):
        import matplotlib.pyplot as plt
        from IPython import embed
        x = np.arange(100)*0.01
        x = x.reshape((-1, self.dim_inputs))
        y = self.evaluate(x)
        acqs,means,sigmas = self.acq_f_info(x)

        x_train = np.concatenate(self.gp.X, 0)[:-1,0].reshape((-1, self.dim_inputs))
        y_train = self.evaluate(x_train)
        plt.figure()
        plt.subplot(311)
        plt.plot(x, y[:,0], label = "objective function dim 0",color = "tab:brown")
        plt.scatter(x_train, y_train[:,0], color = "tab:blue")
        plt.plot(x, means[:,0], color = "tab:blue", label = "prediction 0")
        plt.fill_between(x.reshape((-1)), (means[:,0]-2*sigmas[:,0]).reshape(-1), (means[:,0]+2*sigmas[:,0]).reshape(-1), color = "tab:blue", alpha = 0.5)
        plt.legend(loc = "best")

        plt.subplot(312)

        plt.plot(x, y[:,1], label = "objective function dim 1",color = "tab:orange")
        plt.scatter(x_train, y_train[:,1], color = "tab:red")
        plt.plot(x, means[:,1], color = "tab:red", label = "prediction 1")
        plt.fill_between(x.reshape((-1)), (means[:,1]-2*sigmas[:,1]).reshape(-1), (means[:,1]+2*sigmas[:,1]).reshape(-1), color = "tab:red", alpha = 0.5)
        plt.legend(loc = "best")

        plt.subplot(313)
        plt.plot(x, acqs, label = "entropy term",color = "tab:green")

        plt.legend(loc = "best")
        plt.tight_layout()
        plt.savefig("./" + str(self.iterations)+".png")      
        #embed() 

    def optimize_acq_f(self, n_iter=50, method = "LBFGS"):
        # optimization of aquisition function to get next query point x
        def obj_LBFGS(x):
            return -self.acq_f(x)

        x_tries = np.random.uniform(self.bounds[0, :], self.bounds[1, :],
                                    size=(10000, self.bounds.shape[1]))
        x_seeds = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=(n_iter, self.bounds.shape[1]))
        ys = -obj_LBFGS(x_tries)
        x_max = x_tries[ys.argmax()].reshape((1, -1))
        max_acq = ys.max()
        if(method == "LBFGS"):
            for x_try in x_seeds:
                # Find the minimum of minus the acquisition function
                res = minimize(obj_LBFGS,
                                x_try.reshape(1, -1),
                                bounds=self.reformat_bounds(self.bounds),
                                method="L-BFGS-B")

                # See if success
                if not res.success:
                    continue

                # Store it if better than previous minimum(maximum).
                if max_acq is None or -res.fun[0] > max_acq:
                    x_max = res.x
                    max_acq = -res.fun[0]
        elif(method == "DIRECT"):
            ys = -obj_LBFGS(x_tries)
            x_max = x_tries[ys.argmax()].reshape((1, -1))
            max_acq = ys.max()
            x = scipydirect.minimize(obj_LBFGS, self.reformat_bounds(self.bounds)).x
            acq = -obj_LBFGS(x)[0,0]
            if (acq > max_acq):
                x_max = x
        else:
            raise NotImplementedError

        return np.clip(x_max, self.bounds[0, :], self.bounds[1, :]).reshape((1, -1))

    def acq_f(self, x):
        # aquisition funcition in Maximum Entropy Search
        x = np.reshape(x, (-1, self.paras_gp["input_dim"]))
        means, variances = self.gp.predict(x)
        sigmas = np.sqrt(variances)
        sigmas = sigmas.reshape((-1, self.dim_outputs))
        acq = sigmas.mean(1).reshape((-1,1))
        return acq

    def acq_f_info(self, x):
        #print informations in acquisition funtions: acquistion values, means, sigmas, beta
        x = np.reshape(x, (-1, self.paras_gp["input_dim"]))
        means, variances = self.gp.predict(x)
        sigmas = np.sqrt(variances)
        means = means.reshape((-1, self.dim_outputs))
        sigmas = sigmas.reshape((-1, self.dim_outputs))
        acq = sigmas.mean(1).reshape((-1,1))
        return acq, means, sigmas

    def record(self, x, y):
        pass
        
    def reformat_bounds(self, bounds):
        assert len(bounds) == 2, "unexpected number of bounds"
        return list(zip(*bounds))    



if __name__ == "__main__":

    demo = "GP-UCB"
    if(demo == "GP-UCB"):
        paras_gp={}
        paras_gp["input_dim"] = 2
        paras_gp["noise_variance"] = 0.01**2
        paras_gp["fixed_noise_variance"] = True
        paras_gp["lengthscales"] = np.array([1,1])
        paras_gp["mean_value"] = 0
        paras_gp["kernel_variance"] = 1**2
        paras_gp["bounds"] = np.array([[0,0],[1,1]])

        paras_opt = {}
        paras_opt["num_iterations"] = 100
        paras_opt["num_init_points"] = 5
        paras={}
        paras["gp"] = paras_gp
        paras["opt"] = paras_opt


        def f(x):
            return np.sin(np.pi * x[:,0]) * (0.5 + x[:, 1])

        bo = BO(f, paras)
        bo.optimize()
    elif(demo == "MES"):
        paras_gp={}
        paras_gp["input_dim"] = 2
        paras_gp["output_dim"] = 2
        paras_gp["rank"] = 1
        paras_gp["noise_variance"] = [0.01**2, 0.01**2]
        paras_gp["fixed_noise_variance"] = True
        paras_gp["lengthscales"] = np.array([1,1])
        paras_gp["mean_value"] = [0]
        paras_gp["kernel_variance"] = 1**2
        paras_gp["bounds"] = np.array([[0,0],[1,1]])

        paras_opt = {}
        paras_opt["num_iterations"] = 100
        paras_opt["num_init_points"] = 1
        paras={}
        paras["gp"] = paras_gp
        paras["opt"] = paras_opt

        def f(x):
            return x

        mes = MaximumEntropySearch(f, paras)
        mes.optimize()
    else:
        pass



