import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary, set_trainable, to_default_float
from IPython import embed






'''
for all fucntions to be modeled,
the input is a array with shape (N, input_dim)
the output is a array with shape (N, output_dim)
'''

class GP(object):
    def __init__(self, paras):
        self.input_dim = paras["input_dim"]
        self.mean_value = paras["mean_value"]
        self.fixed_noise_variance = paras["fixed_noise_variance"]
        self.noise_variance = paras["noise_variance"]
        self.lengthscales = paras["lengthscales"]
        self.kernel_variance = paras["kernel_variance"]
        self.X=[]
        self.Y=[]

    def addData(self, x, y):
        self.X.append(x)
        self.Y.append(y)
    def optimizeModel(self):
        k = gpflow.kernels.Matern52(self.kernel_variance, self.lengthscales)
        X = np.concatenate(self.X, 0)
        Y = np.concatenate(self.Y, 0)
        X = X.reshape((-1, self.input_dim))
        Y = Y.reshape((-1, 1))


        meanf = gpflow.mean_functions.Constant(self.mean_value)
        self.gp = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=meanf)
        self.gp.likelihood.variance.assign(self.noise_variance)
        #keep prior mean functions fixed
        #set_trainable(self.gp.mean_function.c, False)
        if(self.fixed_noise_variance):
            set_trainable(self.gp.likelihood.variance, False)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(self.gp.training_loss, self.gp.trainable_variables, options=dict(maxiter=100))
        print_summary(self.gp)

    def predict(self, x):
        means, variances = self.gp.predict_f(x)
        return means.numpy(), variances.numpy()




class MultiOutputGP(object):
    '''
    implementation of LCM GP model
    '''
    def __init__(self, paras):
        self.input_dim = paras["input_dim"]
        self.output_dim = paras["output_dim"]
        self.rank = paras["rank"]
        self.mean_value = paras["mean_value"]
        self.fixed_noise_variance = paras["fixed_noise_variance"]
        self.noise_variance = paras["noise_variance"]
        self.lengthscales = paras["lengthscales"]
        self.kernel_variance = paras["kernel_variance"]
        self.X=[]
        self.Y=[]

    def addData(self, x, y):
        '''
        x: shape:(N, input_dim)
        y: shape:(N, output_dim)
        here resize the data
        '''
        for i in range(self.output_dim):
            self.X.append(np.hstack([x, i * np.ones((x.shape[0],1))]))
            self.Y.append(np.hstack([y[:,i].reshape((-1,1)), i * np.ones((x.shape[0],1))]))

    def optimizeModel(self):
        output_dim = self.output_dim
        rank = self.rank
        k = gpflow.kernels.Matern52(self.kernel_variance, self.lengthscales, active_dims= np.arange(self.input_dim).tolist())
        coreg = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[self.input_dim])
        k = k * coreg
        X = np.concatenate(self.X, 0)
        Y = np.concatenate(self.Y, 0)
        X = X.reshape((X.shape[0], -1))
        Y = Y.reshape((Y.shape[0], -1))
        meanf = gpflow.mean_functions.Constant(self.mean_value)
        lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian() for i in range(self.output_dim)])

        # now build the GP model as normal
        self.gp = gpflow.models.VGP((X,Y), kernel=k, likelihood=lik, mean_function = meanf)
        for i in range(self.output_dim):
            self.gp.likelihood.likelihoods[i].variance.assign(self.noise_variance[i])
        if(self.fixed_noise_variance):
            for i in range(self.output_dim):
                set_trainable(self.gp.likelihood.likelihoods[i].variance, False)
        gpflow.optimizers.Scipy().minimize(self.gp.training_loss, self.gp.trainable_variables, options=dict(maxiter=10000), method="L-BFGS-B")
        print_summary(self.gp)

    def predict(self, x):
        means=[]
        variances=[]
        for i in range(self.output_dim):
            means_dim, variances_dim = self.gp.predict_f(np.hstack([x, i*np.ones((x.shape[0], 1))]))
            means.append(means_dim.numpy())
            variances.append(variances_dim.numpy())
        return np.concatenate(means,1), np.concatenate(variances,1)






if __name__ == "__main__":

    plt.rcParams["figure.figsize"] = (12, 6)


    option = "MOGP"
    if(option == "GP"):
        # example of single output GP
        paras={}
        paras["input_dim"] = 1
        paras["noise_variance"] = 0.01**2
        paras["fixed_noise_variance"] = True
        paras["lengthscales"] = np.array([1])
        paras["mean_value"] = 0
        paras["kernel_variance"] = 1**2
        gp = GP(paras)

        def f(x):
            return x*np.sin(x)
        x = np.arange(100)*0.1
        y = f(x)

        x_train = np.random.rand(10).reshape((-1,1)) * 10
        y_train =  f(x_train).reshape((-1, 1))
        gp.addData(x_train, y_train)
        gp.optimizeModel()


        #plot functions
        plt.plot(x, y, label = "objective function")
        means, variances = gp.predict(x.reshape((-1, 1)))
        plt.plot(x, means, color = "tab:orange", label = "prediction")
        plt.fill_between(x.reshape((-1)), (means-2*np.sqrt(variances)).reshape(-1), (means+2*np.sqrt(variances)).reshape(-1), color = "tab:orange", alpha = 0.5)
        plt.scatter(x_train, y_train)
        plt.legend()
        plt.show()
    else:
        # example of single output GP
        paras={}
        paras["input_dim"] = 1
        paras["output_dim"] = 2
        paras["rank"] = 1
        paras["noise_variance"] = [0.01**2, 0.01**2]
        paras["fixed_noise_variance"] = True
        paras["lengthscales"] = np.array([1])
        paras["mean_value"] = [0]
        paras["kernel_variance"] = 1**2
        gp = MultiOutputGP(paras)

        def f(x):
            return np.concatenate([x * np.sin(2*x), np.sin(6*x + 0.7)],1)
        x = (np.arange(100)*0.01).reshape((-1,1))
        y = f(x)

        x_train = np.random.rand(5).reshape((-1,1))
        y_train =  f(x_train)
        gp.addData(x_train, y_train)
        gp.optimizeModel()


        #plot functions
        plt.plot(x, y[:,0], label = "objective function dim 0",color = "tab:blue")
        plt.plot(x, y[:,1], label = "objective function dim 1", color = "tab:orange")
        means, variances = gp.predict(x.reshape((-1, 1)))

        plt.plot(x, means[:,0], color = "tab:red", label = "prediction 0")
        plt.fill_between(x.reshape((-1)), (means[:,0]-2*np.sqrt(variances[:,0])).reshape(-1), (means[:,0]+2*np.sqrt(variances[:,0])).reshape(-1), color = "tab:red", alpha = 0.5)

        plt.plot(x, means[:,1], color = "tab:green", label = "prediction 1")
        plt.fill_between(x.reshape((-1)), (means[:,1]-2*np.sqrt(variances[:,1])).reshape(-1), (means[:,1]+2*np.sqrt(variances[:,1])).reshape(-1), color = "tab:green", alpha = 0.5)

        plt.scatter(x_train, y_train[:,0], color = "tab:blue")
        plt.scatter(x_train, y_train[:,1], color = "tab:orange")
        plt.legend()
        plt.show()
