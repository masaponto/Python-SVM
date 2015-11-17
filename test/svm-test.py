#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import sys
sys.path.append('../')

import svm


class SVMTest(unittest.TestCase):

    def setUp(self):
        self.svm = svm.SVM()

    def test_gausiian_kernel(self):
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])

        self.assertEqual(self.svm._gaussian_kernel(a, b), 0.22313016014842987)

        #from sklearn.metrics import pairwise
        #self.assertEqual(self.svm._gaussian_kernel(a, b), pairwise.rbf_kernel(a, b, gamma = 1/2))




if __name__ == "__main__":
    unittest.main()
