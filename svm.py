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

    def __init__(self,
                 c=10.0,
                 loop = 1000,
                 tol = 1e-2,
                 eps = 1e-2):

        self._c = c
        self._loop = loop
        self._tol = tol
        self._eps = eps

    def _gaussian_kernel(self, x_v1, x_v2, delta=1.0):
        return np.exp(- (np.linalg.norm(x_v1 - x_v2) ** 2) / (2 * delta**2))

    def _take_step(self, i1, i2):

        if i1 == i2:
            return False

        alph1 = self._alpha[i1]
        alph2 = self._alpha[i2]
        y1 = self._y[i1]
        y2 = self._y[i2]
        e1 = self._e[i1]
        e2 = self._e[i2]
        s = y1 * y2

        if y1 != y2:
            L = max(0, alph2 - alph1)
            H = min(self._c, self._c - alph1 + alph2)
        else:
            L = max(0, alph2 + alph1 - self._c)
            H = min(self._c, alph1 + alph2)

        if L == H:
            return False

        k11 = self._gaussian_kernel(self._x_vs[i1], self._x_vs[i1])
        k12 = self._gaussian_kernel(self._x_vs[i1], self._x_vs[i2])
        k22 = self._gaussian_kernel(self._x_vs[i2], self._x_vs[i2])

        # why
        eta = 2 * k12 - k11 - k22

        #if eta > 0:
        #    print('eta > 0')
        #    return False

        if eta < 0:
            print('eta =',eta)
            # 一点目の更新処理とクリッピング
            print('一点目の更新処理とクリッピング')
            a2 = alph2 - y2 * (e1 - e2) / eta
            print('y2 * (e1 - e2) / eta', y2 * (e1 - e2) / eta)

            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            print('カーネルが正定値でない場合の処理')
            # カーネルが正定値でない場合の処理
            a1 = self._alpha[i1]
            a2 = self._alpha[i2]
            v1 = self.one_predict(
                self._x_vs[i1]) - self._b - y1 * a1 * k11 - y2 * a2 * k12
            v2 = self.one_predict(
                self._x_vs[i2]) - self._b - y1 * a1 * k12 - y2 * a2 * k22
            Wconst = 0
            for i in range(n):
                if i != i1 and i != i2:
                    Wconst += a[i1]
            for i in range(n):
                for j in range(n):
                    if i != i1 and i != i2 and j != i1 and j != i2:
                        Wconst += self.y[i] * self.y[j] * \
                            self._gaussian_kernel(self._x_vs[i], self._x_vs[j]) * a[i] * a[j] / 2.0

            a2 = L
            a1 = y1 * self._alpha[i1] + y2 * self._alpha[i2] - y2 * L
            Lobj = a1 + a2 - k11 * a1 * a1 / 2 - k22 * a2 * a2 / 2 - s * \
                k12 * a1 * a2 / 2 - y1 * a1 * v1 - y2 * a2 * v2 + Wconst

            a2 = H
            a1 = y1 * self._alpha[i1] + y2 * self._alpha[i2] - y2 * H
            Hobj = a1 + a2 - k11 * a1 * a1 / 2 - k22 * a2 * a2 / 2 - s * \
                k12 * a1 * a2 / 2 - y1 * a1 * v1 - y2 * a2 * v2 + Wconst

            if Lobj > Hobj + self._eps:
                a2 = L
            elif Lobj > Hobj - self._eps:
                a2 = H
            else:
                a2 = alph2

        if a2 < 1e-8:
            a2 = 0
        elif a2 > self._c - 1e-8:
            a2 = self._c

        if abs(a2 - alph2) < self._eps * (a2 + alph2 + self._eps):
            print('a2 =', a2, ' alph2 =', alph2)
            print('abs(a2 - alph2) = ', abs(a2 - alph2))
            return False

        a1 = alph1 + s * (alph2 - a2)

        b_old = self._b
        b1 = e1 + y1 * (a1 - self._alpha[i1]) * \
            k11 + y2 * (a2 - self._alpha[i2]) * k12 + self._b
        b2 = e2 + y1 * (a1 - self._alpha[i1]) * \
            k12 + y2 * (a2 - self._alpha[i2]) * k22 + self._b

        if b1 == b2:
            self._b = b1
        else:
            self._b = (b1 + b2) / 2

        # update error chache using new lagrange multipliers

        da1 = a1 - self._alpha[i1]
        da2 = a2 - self._alpha[i2]

        for i in range(len(self._x_vs)):
            self._e[i] = self._e[i] + y1 * da1 * \
                self._gaussian_kernel(self._x_vs[i1], self._x_vs[i]) + y2 * da2 * \
                self._gaussian_kernel(self._x_vs[i2], self._x_vs[i]) + b_old - self._b

        # store a1, a2 in the alpha array
        self._alpha[i1] = a1
        self._alpha[i2] = a2

        return True

    def _search(self, i, lst):
        if self._e[i] >= 0:
            return reduce(lambda j, k : j if self._e[j] < self._e[k] else k, lst)
        else:
            return reduce(lambda j, k: j if self._e[j] > self._e[k] else k, lst)


    def _examine_example(self, i2):
        y2 = self._y[i2]
        alph2 = self._alpha[i2]
        e2 = self._e[i2]
        r2 = e2 * y2
        i1 = 0

        if (r2 < -self._tol and alph2 < self._c) or (r2 > self._tol and alph2 > 0):
            alst1 = [i for i in range(len(self._alpha))
                     if 0 <self._alpha[i] < self._c]

            if alst1:
                i1 = self._search(i2, alst1)
                if self._take_step(i1, i2):
                    return True

                random.shuffle(alst1)

                for i1 in alst1:
                    if self._take_step(i1, i2):
                        return True

            alst2 = [i for i in range(len(self._alpha))
                     if (self._alpha[i] <= 0 or
                         self._alpha[i] >= self._c)]

            random.shuffle(alst2)

            for i1 in alst2:
                if self._take_step(i1, i2):
                    return True

            self._calc_b()

        return False

    def _calc_b(self):
        self._s = [i for i in range(len(self._y))
                   if 0 < self._alpha[i]]
        self._m = [i for i in range(len(self._y))
                   if 0 < self._alpha[i] < self._c]

        self._b = 0.0
        for i in self._m:
            self._b += self._y[i]
            for j in self._s:
                self._b -= (self._alpha[j]*self._y[j]*self._gaussian_kernel(self._x_vs[i], self._x_vs[j]))

        self._b /= len(self._m)


    def fit(self, X, y):
        self._y = y
        self._x_vs = X
        self._alpha = np.zeros(len(y), dtype=float)
        self._b = 0
        self._e = -1 * np.array(y, dtype=float)
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
                for i in range(len(self._y)):
                    changed |= self._examine_example(i)

            if examine_all:
                examine_all = False

            elif not changed:
                examine_all = True

        print('hoo')
        self._calc_b()

    def one_predict(self, x_v):
        tmp = 0
        for i in range(len(self._x_vs)):
            tmp += self._alpha[i] * self._y[i] * _gaussian_kernel(x_v, self._x_vs[i])
        return tmp - self._b



def main():

    svm = SVM(c = 50)

    db_name = 'australian'
    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data_set.data, data_set.target, test_size=0.4, random_state=0)

    svm.fit(X_train, y_train)

    re = np.array([ svm.one_predict(x) for x in X_test])

    print(sum([r == y for r, y in zip(re, y_test)]) / len(y_test))


if __name__ == "__main__":
    main()
