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
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from functools import reduce


class SVM (BaseEstimator):
    """
    Support Vector Machine
    using SMO Algorithm.
    """

    def __init__(self,
                 kernel=lambda x,y:np.dot(x,y),
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
        self._kernel = kernel
        self._c = c
        self._tol = tol
        self._eps = eps
        self._loop = loop


    def _takeStep(self, i1, i2):
        if i1 == i2:
            return False
        alph1 = self._alpha[i1]
        alph2 = self._alpha[i2]
        y1 = self._target[i1]
        y2 = self._target[i2]
        e1 = self._e[i1]
        e2 = self._e[i2]
        s = y1 * y2

        if y1 != y2:
            L = max(0, alph2 - alph1)
            H = min(self._c, self._c-alph1+alph2)
        else:
            L = max(0, alph2 + alph1 - self._c)
            H = min(self._c, alph1+alph2)

        if L == H:
            return False

        k11 = self._kernel(self._point[i1], self._point[i1])
        k12 = self._kernel(self._point[i1], self._point[i2])
        k22 = self._kernel(self._point[i2], self._point[i2])
        eta = 2 * k12 - k11 - k22
        if eta > 0:
            return False

        a2 = alph2 - y2 * (e1 - e2) / eta

        a2 = min(H, max(a2, L))

        if abs(a2 - alph2) < self._eps * (a2 + alph2 + self._eps):
            return False
        a1 = alph1 + s * (alph2 - a2)

        # update
        da1 = a1 - alph1
        da2 = a2 - alph2

        self._e += np.array([(da1 * self._target[i1] * self._kernel(self._point[i1], p) +
                           da2 * self._target[i2] * self._kernel(self._point[i2], p))
                          for p in self._point])

        self._alpha[i1] = a1
        self._alpha[i2] = a2
        return True

    def _search(self, i, lst):
        if self._e[i] >= 0:
            return reduce(lambda j,k: j if self._e[j] < self._e[k] else k, lst)
        else:
            return reduce(lambda j,k: j if self._e[j] > self._e[k] else k, lst)

    def _examinEx(self, i2):
        y2 = self._target[i2]
        alph2 = self._alpha[i2]
        e2 = self._e[i2]
        r2 = e2*y2
        if ((r2 < -self._tol and alph2 < self._c) or
            (r2 > self._tol and alph2 > 0)):
            alst1 = [i for i in range(len(self._alpha))
                     if 0 < self._alpha[i] < self._c]
            if alst1:
                i1 = self._search(i2, alst1)
                if self._takeStep(i1, i2):
                    return True
                random.shuffle(alst1)
                for i1 in alst1:
                    if self._takeStep(i1, i2):
                        return True

            alst2 = [i for i in range(len(self._alpha))
                     if (self._alpha[i] <= 0 or
                         self._alpha[i] >= self._c)]

            random.shuffle(alst2)
            for i1 in alst2:
                if self._takeStep(i1, i2):
                    return True

            self._calc_b()

        return False

    def _calc_b(self):
        self._s = [i for i in range(len(self._target))
                   if 0 < self._alpha[i]]
        self._m = [i for i in range(len(self._target))
                   if 0 < self._alpha[i] < self._c]
        self._b = 0.0
        for i in self._m:
            self._b += self._target[i]
            for j in self._s:
                self._b -= (self._alpha[j]*self._target[j]*
                            self._kernel(self._point[i], self._point[j]))
        self._b /= len(self._m)

    def one_predict(self, x):
        ret = self._b
        for i in self._s:
            ret += (self._alpha[i]*self._target[i]*
                    self._kernel(x, self._point[i]))
        return np.sign(ret)

    def fit(self, point, target):
        self._target = target
        self._point = point

        self._alpha = np.zeros(len(target), dtype=float)
        self._b = 0
        self._e = -1*np.array(target, dtype=float)
        changed = False
        examine_all = True
        count = 0

        while changed or examine_all:
            count += 1
            print(count)
            if count > self._loop:
                break

            changed = False

            if examine_all:
                for i in range(len(self._target)):
                    changed |= self._examinEx(i)
            else:
                for i in (j for j in range(len(self._target))
                          if 0 < self._alpha[j] < self._c):
                    changed |= self._examinEx(i)

            if examine_all:
                examine_all = False
            elif not changed:
                examine_all = True

        self._calc_b()

def main():

    svm = SVM(c=100, loop = 20)

    db_name = 'australian'
    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data_set.data, data_set.target, test_size=0.4, random_state=0)

    svm.fit(X_train, y_train)

    re = np.array([svm.one_predict(x) for x in X_test])

    print(re)

    print(sum([r == y for r, y in zip(re, y_test)]) / len(y_test))


if __name__ == "__main__":
    main()
