# -*- coding:utf-8 -*-
import numpy as np
import random as rd
class Discriminant:
    data = []
    trainx = []
    trainy = []
    testx = []
    testy = []
    class_k = None  # the number of classes
    def __init__(self, file, K):
        self.class_k = K
        self.readFile(file)
        self.datalen = len(self.data)
        # print(self.datalen)
        self.splitData()

    def readFile(self, file):
        with open(file, 'r') as fo:
            for row in fo.readlines():
                lines = row.split(',')
                #  4 features and one label(1-3)
                for i in range(len(lines)):
                    lines[i] = float(lines[i])
                if lines[-1] > self.class_k:
                    continue
                else:
                    self.data.append(lines)

    def splitData(self):
        l = 0.6*self.datalen
        print("the number of train set:", l)
        random_list = np.arange(self.datalen)
        rd.shuffle(random_list)
        # print(random_list)
        K = 0
        for i in range(self.datalen):
            if K <= self.data[random_list[i]][-1]:
                K = self.data[random_list[i]][-1]
            if i < l:
                self.trainx.append(self.data[random_list[i]][0:4])
                if self.class_k == 2:
                    self.trainy.append(1 if self.data[random_list[i]][-1]-self.class_k >= 0 else -1)
                else:
                    self.trainy.append(self.data[random_list[i]][-1])
            else:
                self.testx.append(self.data[random_list[i]][0:4])
                if self.class_k == 2:
                    self.testy.append(1 if self.data[random_list[i]][-1]-self.class_k >= 0 else -1)
                else:
                    self.testy.append(self.data[random_list[i]][-1])
        if self.class_k >= K:
            self.class_k = int(K)

    def getDataSplit(self):
        return self.trainx, self.trainy, self.testx, self.testy

    def basisFunction(self, x):
        return x

    def activationFunction(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def perceptronTrainAlgorithm(self, eta, w0, X, Y):
        """
        search decision boundary for two classes
        :param X: N*M
        :param Y: 1
        :return: w(M*1)
        """
        if self.class_k > 2:
            print("this algorithm used for second classification")
            return None

        for i in range(len(X)):
            """
            the first class is denoted to -1
        and the second class is +1
            """
            w = w0 + eta * np.mat(self.basisFunction(X[i])) * Y[i]
            # print((w - w0))
            if np.sqrt(np.sum((w - w0) * (w - w0).T)) < 1e-5:
                print("converge")
                return w
            w0 = w
        print("run for the total data")
        return w

    def perceptronTestAlgorithm(self, w, X):
        """
        precision the label of X
        :param w: M*1
        :param X: N*M
        :return: Y(N*M)
        """
        if self.class_k > 2:
            print("this algorithm used for second classification")
            return None

        Y = []
        for i in range(len(X)):
            y = self.activationFunction(np.mat(X[i])*w.T)
            Y.append(y)
        return Y

    def getMeanData(self, X):
        print(X)
        mean = np.zeros_like(X[0])
        for x in X:
            mean += x
        return mean/len(X), len(X)

    def fisherTrainAlgorithm(self, X, Y):
        """
        search the space for mapping
        :param X:N*M
        :param Y:N*1
        :return:
        """
        if self.class_k > 2:
            print("this algorithm used for second classification")
            return None

        class1 = []
        class2 = []
        for i in range(len(Y)):
            # print(X[i], Y[i])
            if Y[i] < 0:
                class1.append(X[i])
            else:
                class2.append(X[i])

        m1, n1 = self.getMeanData(class1)
        m2, n2 = self.getMeanData(class2)

        x, y = np.array(X).shape
        Sw = np.mat(np.zeros((y, y)))
        for i in range(len(Y)):
            if Y[i] < 0:
                Sw += np.mat(X[i]-m1).T*np.mat(X[i]-m1)
            else:
                Sw += np.mat(X[i]-m2).T*np.mat(X[i]-m2)

        w = Sw.I * np.mat(m2 - m1).T
        m = (n2*m2 + n1*m1) / (n1+n2)

        return w, m

    def fisherTestAlgorithm(self, w, m, X):
        """
        classification for second classes
        :param w:
        :param m:
        :param X:
        :return:
        """
        if self.class_k > 2:
            print("this algorithm used for second classification")
            return None

        n = len(X)
        Y = []
        for i in range(n):
            y = (X[i]-m)*w
            Y.append(self.activationFunction(y))
        return Y

    def fisherMultipleTrainAlgorthm(self, X, Y, d):
        """
        fisher's discriminant for multiple classes
        :param X:
        :param Y:
        :return:
        """
        label = {}
        # print(self.class_k)
        for y in Y:
            label[y] = label.get(y, y)
            if len(label) == self.class_k:
                break
        # print(label)
        classX = []  # the classification data set: K*N*M
        classY = []  # the classes label: K*N*1
        mk = []  # the mean for each classes: K*1*M
        m = np.zeros_like(X[0])  # the mean for total data: 1*M
        nk = []  # the number of each classes: K*1
        for i in label.keys():
            class_x = []
            class_y = []
            m_k = np.zeros_like(X[0])
            n_k = 0
            for j in range(len(X)):
                if Y[j] == label[i]:
                    class_x.append(X[j])    # N*M
                    class_y.append(Y[j])    # N*1
                    m_k += X[j]    # 1*M
                    m += X[j]   # 1*M
                    n_k += 1    # 1*1
            classX.append(class_x)
            classY.append(class_y)
            mk.append(m_k)
            nk.append(n_k)
        # print(classX)
        classX = np.mat(classX[0])
        # print(classX)
        mk = np.mat(mk)
        # print(mk)
        m = np.mat(m)
        l = len(X[0])
        Sw = np.mat(np.zeros((l, l)))
        Sb = Sw

        for i in range(self.class_k):
            Sk = (classX[i]-mk[i]).T * (classX[i]-mk[i])
            Sw += Sk
            Sb += nk[i] * (mk[i]-m).T * (mk[i]-m)
        W = Sw.I * Sb
        # get the eigenvectors of W
        a, b = np.linalg.eig(W)
        arg = a.argsort()
        eigenvalues = []
        eigenvectors = []
        for ii in range(d):
            i = arg[ii]
            eigenvalues.append(a[i])
            eigenvectors.append(np.array(b[i])[0])
        W = np.mat(eigenvectors)
        print(type(W))
        print(type(mk))
        return W, mk, Sb

    def discriminate(self, W, m, Sb, x):
        """

        :param W: D'(d)*M
        :param m: K*M
        :param Sb: M*M
        :param x: 1*M
        :return:
        """
        mk = np.array(m)
        # print(W)
        # sb = W*Sb*W.T
        jw = np.inf
        result = 0
        # print()
        for i in range(self.class_k):
            # d*m * m*1 * 1*m *m*d
            # sw = (W*(np.mat(x) - mk[i]).T) * (np.mat(x) - mk[i])*W.T
            # a, b = np.linalg.eig(sw.I * sb)
            # print(sw)
            # break
            # # print(i)
            a = (np.mat(x) - mk[i])*W.T * (W*(np.mat(x) - mk[i]).T)
            print(a)
            j_w = np.sum(a)
            if j_w <= jw:
                result = i+1
                jw = j_w
        return result


    def fisherMultipleTestAlgorthm(self, W, mk, Sb, X):
        Y = []
        i = 0
        for x in X:
            # print(i)
            y = self.discriminate(W, mk, Sb, x)
            Y.append(y)
            i += 1
            # break
        return Y

    def evaluation(self, real, precision):
        """
        compute accuracy
        :param real: the real sequence
        :param precision: we precise sequence
        :return: the accuracy
        """
        l = len(precision)
        tp = 0
        for i in range(l):
            if precision[i] == real[i]:
                tp += 1
        return tp/l

if __name__ == '__main__':
    # N:100 M:4
    filepath = 'E:\PRML\Classification\iris.data.txt'

    K = 4
    M = Discriminant(filepath, K)
    trainx, trainy, testx, testy = M.getDataSplit()

    # # perceptron Algorithm
    # w0 = np.mat([1, 1, 1, 1])
    # eta = 1
    # w = M.perceptronTrainAlgorithm(eta, w0, trainx, trainy)
    # print("the precision boundary:", w)
    # Y = M.perceptronTestAlgorithm(w, testx)
    # tp = M.evaluation(testy, Y)
    # print(tp)

    # # fisher's discrimination
    w, m = M.fisherTrainAlgorithm(trainx, trainy)
    Y = M.fisherTestAlgorithm(w, m, testx)
    tp = M.evaluation(testy, Y)
    print(tp)

    # multiple classes
    # W, mk, Sb = M.fisherMultipleTrainAlgorthm(trainx, trainy, 3)
    # Y = M.fisherMultipleTestAlgorthm(W, mk, Sb, testx)
    # print(testy)
    # print(Y)