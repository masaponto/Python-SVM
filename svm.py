#!/usr/bin/env python
# -*- coding: utf-8 -*-

# reference
# http://www.slideshare.net/sleepy_yoshi/smo-svm
# http://courses.cs.tau.ac.il/0368-4341/shared/Papers/SMO/smo-book%20+%20pseudocode.pdf
# http://kivantium.hateblo.jp/entry/2015/06/24/145734
# http://sage.math.canterbury.ac.nz/home/pub/105/


import numpy as np
import random

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
from functools import reduce


class SVM(BaseEstimator):
    """
    Support Vector Machine
    using SMO Algorithm.
    """

    def __init__(self,
                 kernels=lambda x,y:np.dot(x,y),
                 c=10000,
                 tol=1e-2,
                 eps=1e-2,
                 loop=float('inf')):
        """
        Arguments:
        - `kernel`: カーネル関数
        - `c`: パラメータ
        - `tol`: KKT条件の許容する誤差
        - `eps`: αの許容する誤差
        - `loop`: ループの上限
        """
        self.kernels = kernels
        self.c = c
        self.tol = tol
        self.eps = eps
        self.loop = loop


    def _takeStep(self, i1, i2):

        if i1 == i2:
            return False

        alph1 = self.alpha[i1]
        alph2 = self.alpha[i2]
        y1 = self.target[i1]
        y2 = self.target[i2]
        e1 = self.e[i1]
        e2 = self.e[i2]
        s = y1 * y2

        if y1 != y2:
            L = max(0, alph2 - alph1)
            H = min(self.c, self.c-alph1+alph2)
        else:
            L = max(0, alph2 + alph1 - self.c)
            H = min(self.c, alph1+alph2)

        if L == H:
            return False

        k11 = self.kernels(self.point[i1], self.point[i1])
        k12 = self.kernels(self.point[i1], self.point[i2])
        k22 = self.kernels(self.point[i2], self.point[i2])
        eta = 2 * k12 - k11 - k22
        if eta > 0:
            return False

        a2 = alph2 - y2 * (e1 - e2) / eta

        a2 = min(H, max(a2, L))

        if abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
            return False

        a1 = alph1 + s * (alph2 - a2)

        # update
        da1 = a1 - alph1
        da2 = a2 - alph2

        self.e += np.array([(da1 * self.target[i1] * self.kernels(self.point[i1], p) +
                           da2 * self.target[i2] * self.kernels(self.point[i2], p))
                          for p in self.point])

        self.alpha[i1] = a1
        self.alpha[i2] = a2

        return True

    def _search(self, i, lst):
        if self.e[i] >= 0:
            return reduce(lambda j,k: j if self.e[j] < self.e[k] else k, lst)
        else:
            return reduce(lambda j,k: j if self.e[j] > self.e[k] else k, lst)

    def _examinEx(self, i2):
        y2 = self.target[i2]
        alph2 = self.alpha[i2]
        e2 = self.e[i2]
        r2 = e2*y2
        if ((r2 < -self.tol and alph2 < self.c) or
            (r2 > self.tol and alph2 > 0)):
            alst1 = [i for i in range(len(self.alpha))
                     if 0 < self.alpha[i] < self.c]
            if alst1:
                i1 = self._search(i2, alst1)
                if self._takeStep(i1, i2):
                    return True
                random.shuffle(alst1)
                for i1 in alst1:
                    if self._takeStep(i1, i2):
                        return True

            alst2 = [i for i in range(len(self.alpha))
                     if (self.alpha[i] <= 0 or
                         self.alpha[i] >= self.c)]

            random.shuffle(alst2)
            for i1 in alst2:
                if self._takeStep(i1, i2):
                    return True

            self._calc_b()

        return False

    def _calc_b(self):
        self.s = [i for i in range(len(self.target))
                   if 0 < self.alpha[i]]
        self.m = [i for i in range(len(self.target))
                   if 0 < self.alpha[i] < self.c]
        self.b = 0.0
        for i in self.m:
            self.b += self.target[i]
            for j in self.s:
                self.b -= (self.alpha[j]*self.target[j]*
                            self.kernels(self.point[i], self.point[j]))
        self.b /= len(self.m)

    def one_predict(self, x):
        ret = self.b
        for i in self.s:
            ret += (self.alpha[i]*self.target[i]*
                    self.kernels(x, self.point[i]))
        return np.sign(ret)

    def predict(self, X):
        return np.array(list(map(self.one_predict, X)))

    def fit(self, X, y):
        self.target = y
        self.point = X

        self.alpha = np.zeros(len(y), dtype=float)
        self.b = 0
        self.e = -1*np.array(y, dtype=float)
        changed = False
        examine_all = True
        count = 0

        while changed or examine_all:
            count += 1

            if count > self.loop:
                break

            changed = False

            if examine_all:
                for i in range(len(self.target)):
                    changed |= self._examinEx(i)
            else:
                for i in (j for j in range(len(self.target))
                          if 0 < self.alpha[j] < self.c):
                    changed |= self._examinEx(i)

            if examine_all:
                examine_all = False
            elif not changed:
                examine_all = True

        self._calc_b()


def cross_val():
    db_names = ['australian']

    for db_name in db_names:
        print(db_name)
        data_set = fetch_mldata(db_name)
        data_set.data = preprocessing.scale(data_set.data)
        svm = SVM(c=100, loop=20)
        scores = cross_validation.cross_val_score(svm, data_set.data, data_set.target, cv=5, scoring='accuracy')
        print("Accuracy: %0.3f " % (scores.mean()))


def test():
    svm = SVM(c=100, loop=20)

    db_name = 'australian'
    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data_set.data, data_set.target, test_size=0.4, random_state=0)

    svm.fit(X_train, y_train)

    re = svm.predict(X_test)

    print("Accuracy: %0.3f " % (sum([r == y for r, y in zip(re, y_test)]) / len(y_test) ))


def main():
    cross_val()
    #test()

if __name__ == "__main__":
    main()
