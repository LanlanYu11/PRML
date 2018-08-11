#   -*- coding=utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt

class BiasVariance:
    """
    use maximum likelihood function and least squares
    for Linear basis function models
    """
    file = None
    trainx = []
    trainy = []
    testx = []
    testy = []
    regularCoeff = 0
    w = 0
    K = 20
    def __init__(self, file, regularCoeff, k ):
        """
        init function
        :param file: read data from file
        """
        self.file = file    # file path
        self.regularCoeff = regularCoeff    # regularity coefficient
        self.K = k
        self.splitData()    # get test set and train set
        print("read successfully. ")

        l = len(self.trainy)
        # print(l)
        """
        sequence learning
        imitate the method of figure3.5
        """
        # W = np.mat(np.zeros((self.K, 1)))
        # print(W)
        # L = 0
        # trainx = []
        # trainy = []
        # for i in range(l):
        #     trainy.append(self.trainy[i])
        #     trainx.append(self.trainx[i])
        #     if (i+1)%20 == 0:   # 样本条目满20个
        #         w = self.wFunction(trainx, trainy)
        #         # print(i+1)
        #         W += w
        #         L += 1
        #         # trainx = []
        #         # trainy = []
        # w = self.wFunction(trainx, trainy)
        # W += w
        # L += 1
        # self.w = W/L
        """
        total data
        the likelihood function
        """
        self.w = self.wFunction(self.trainx, self.trainy)


    def getW(self):
        """
        get the value of parameter w
        :return: parameter w
        """
        return self.w

    def readFile(self):
        """
        :input : file read data
        :output: the data list read from file
        """
        with open(self.file, 'r', encoding='utf-8') as read:
            data = []
            for lines in read.readlines()[3:]:
                line = lines.split()
                line[0] = float(line[0])
                line[1] = float(line[1])
                data.append(line)
        # print(data)
        return data

    def splitData(self):
        """
        :input: divide data into train set and test set by Leave-One Method
        :return: train set and test set
        """
        data = self.readFile()
        s = len(data)
        i = l = int(0.4*s)
        print(i)
        for j in range(s):
            if len(self.trainx) == (s-l):
                break
            d = data[j]
            if len(self.testx) < l:
                if random.random() < 1e-1 and i > 0:
                    print(i)
                    self.testx.append(d[0])
                    self.testy.append(d[-1])
                    i -= 1
                    continue
            self.trainx.append(d[0])
            self.trainy.append(d[-1])
        data.reverse()
        for d in data:
            if l == len(self.testx):
                break
            self.testx.append(d[0])
            self.testy.append(d[-1])
            print(i)
            i -= 1

    def basisFuction(self, x):
        """
        compute basis function, Gaussian
        :param x: the data
        :return: the result of this data by Gaussian
        """
        F = [1]
        for i in range(self.K-1):
            i += 1
            f = np.exp(-0.5*((x-1*i)*(x-1*i))/(1.3*1.3))
            F.append(f)
        return F

    def wFunction(self, trainx, trainy):
        """
        :param trainx: train data
        :param trainy: the real target of train set
        :return: the parameter of function
        """
        n = len(trainx)
        I = np.diag(np.ones(self.K))
        # print("the size of one sample: ", I)
        F = []
        for i in range(n):
            F.append(self.basisFuction(trainx[i]))
        F = np.mat(F)
        # print(F)
        # print((F.T*F))
        # print(np.mat(trainy).T)
        w = np.mat(self.regularCoeff * I + F.T*F).I * np.mat(F.T) * np.mat(trainy).T
        print(w)
        return w

    def compute(self, X):
        """
        compute the prediction target
        :param X:  data
        :return: target
        """
        T = []
        for x in X:
            F = np.mat(self.basisFuction(x))
            T.append((F * self.w)[0, 0])
            # T.append(self.w[0, 0] + self.w[1, 0]*self.basisFuction(x))
        return T

    def draw(self):
        """
        draw the real data and prediction data
        """
        X = self.testx
        X.sort()
        Y = self.testy
        T = self.compute(X)
        print(X, Y, T)
        plt.figure()
        plt.plot(X, Y, color='g')
        plt.plot(X, T, color='r')
        plt.show()



class Bayesian:
    """
    use maximizing the evidence function
    for Bayesian linear regression
    """
    file = None
    trainx = []
    trainy = []
    testx = []
    testy = []
    K = 20
    alpha = None
    beta = None
    mN = None
    sN = None

    def __init__(self, file, k, alpha, beta):
        """
        init function
        :param file: read data from file
        :param k: the number of basis function
        """
        self.file = file  # file path
        self.K = k
        self.splitData()  # get test set and train set
        print("read successfully. ")

        l = len(self.trainy)

        beta = beta
        alpha = alpha

        trainx = []
        trainy = []
        for i in range(l):
            trainy.append(self.trainy[i])
            trainx.append(self.trainx[i])
            if (i + 1) % 20 == 0:  # 样本条目满20个
                alpha, beta = self.update(alpha, beta, trainx, trainy)
                print("turn: ", i, alpha, beta)
                # trainx = []
                # trainy = []
        self.alpha, self.beta = self.update(alpha, beta, trainx, trainy)
        self.mN, self.sN = self.mN_sN(alpha, beta, trainx, trainy)

    def readFile(self):
        """
        :input : file read data
        :output: the data list read from file
        """
        with open(self.file, 'r', encoding='utf-8') as read:
            data = []
            for lines in read.readlines()[3:]:
                line = lines.split()
                line[0] = float(line[0])
                line[1] = float(line[1])
                data.append(line)
        # print(data)
        return data

    def splitData(self):
        """
        :input: divide data into train set and test set by Leave-One Method
        :return: train set and test set
        """
        data = self.readFile()
        s = len(data)
        i = l = int(0.4*s)
        print(i)
        for j in range(s):
            if len(self.trainx) == (s-l):
                break
            d = data[j]
            if len(self.testx) < l:
                if random.random() < 1e-1 and i > 0:
                    print(i)
                    self.testx.append(d[0])
                    self.testy.append(d[-1])
                    i -= 1
                    continue
            self.trainx.append(d[0])
            self.trainy.append(d[-1])
        data.reverse()
        for d in data:
            if l == len(self.testx):
                break
            self.testx.append(d[0])
            self.testy.append(d[-1])
            print(i)
            i -= 1

    def basisFunction(self, x):
        """
        compute basis function, Gaussian
        :param x: the data
        :return: the result of this data by Gaussian 1*M
        """
        F = [1]
        for i in range(self.K - 1):
            i += 1
            f = np.exp(-0.5 * ((x - 1 * i) * (x - 1 * i)) / (1.3 * 1.3))
            F.append(f)
        return F

    def beta(self, gamma, X, T, m):
        """
        value
        :param gamma:
        :param X:1*N
        :param T: 1*N
        :param m: M*1
        :return:
        """
        beta = 0
        l = len(X)
        for i in range(l):
            beta += (T[i] - (m.T*np.mat(self.basisFunction(X[i])).T)[0,0])**2
        return beta/(l-gamma)

    def alpha(self, gamma, m):
        """
        value
        :param gamma:
        :param m: M*1
        :return:
        """
        alpha = gamma/(m.T*m)[0, 0]
        return alpha

    def getFmat(self, X):
        """
        shape: N * M
        :param X:
        :return:
        """
        F = []
        for x in X:
            f = self.basisFunction(x)
            F.append(f)
        return np.mat(F)

    def mN(self, alpha, beta, X, T):
        """
        shape: M * 1
        :param alpha:
        :param beta:
        :param X:
        :param T:
        :return:
        """
        l = len(T)
        M = self.getFmat(X)
        mN = beta * np.matmul(np.linalg.inv(alpha*np.eye(self.K) + beta*M.T*M), self.getFmat(X).T) * np.mat(T).T
        return mN

    def gamma(self, alpha, beta, X):
        """
        sum value
        :param alpha:
        :param beta:
        :param X:
        :return:
        """
        M = self.getFmat(X)
        values= np.linalg.eigvals(beta*M.T*M)
        gamma = 0
        for v in values:
            gamma += v/(v+alpha)
        return gamma

    def update(self, alpha, beta, X, T):
        """
        update the alpha and beta value
        :param alpha:
        :param beta:
        :param X:
        :param T:
        :return:
        """
        mN = self.mN(alpha, beta, X, T)
        gamma = self.gamma(alpha, beta, X)
        print(mN.shape)
        alpha = self.alpha(gamma, mN)
        beta = self.beta(gamma, X, T, mN)
        return alpha, beta
    def mN_sN(self, alpha, beta, X, T):
        """
        compute the final mN and sN
        :param alpha:
        :param beta:
        :param X:
        :param T:
        :return mN: M*1
        :return sM: M*M
        """
        l = len(X)
        mN = self.mN(alpha, beta, X, T)
        M = self.getFmat(X)     # shape: N*M
        sN = np.linalg.inv(alpha * np.eye(self.K) + beta * M.T * M)
        return mN, sN

    def compute(self, X):
        """
        mean is the prediction value
        :param x:
        :param alpha:
        :param beta:
        :param mN:  M*1
        :param sN:  M*M
        :return:
        """
        T = []
        for x in X:
            bx = self.basisFunction(x)
            mean = self.mN.T * np.mat(bx).T
            var = 1/self.beta + np.mat(bx) * self.sN * np.mat(bx).T
            t = mean[0, 0]
            T.append(t)
        return T
    def draw(self):
        """
        draw the real data and prediction data
        """
        X = self.testx
        X.sort()
        Y = self.testy
        T = self.compute(X)
        print(X, Y, T)
        plt.figure()
        plt.plot(X, Y, color='g')
        plt.plot(X, T, color='r')
        plt.show()

if __name__ == "__main__":
    file = ".\data.txt"

    """
    set the regularization coefficient for controlling the relative importance -
    -of the data-dependent error and the regularization term
    """
    lamb = np.exp(-2.4)
    """
    the number of basis function
    """
    # m = 13
    # print(np.exp(-2.4))
    # M = BiasVariance(file, lamb, m)
    # w = M.getW()
    # print(w)
    # M.draw()

    """
    make the initial choice for α and β
    """
    alpha = 0.005
    beta = 25
    """
        the number of basis function
    """
    m = 27
    M = Bayesian(file, m, alpha, beta)
    M.draw()
