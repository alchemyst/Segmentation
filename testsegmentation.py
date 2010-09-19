#!/usr/bin/env python

import unittest
import segment
import numpy

class TestDataContainer(unittest.TestCase):
    def setUp(self):
        self.x = numpy.array([1, 2, 3])
        self.y = numpy.array([4, 5, 6])

    def testinit(self):
        """construction from x and y data"""
        a = segment.DataContainer(self.x, self.y)
        assert numpy.all(self.x == a.x)
        assert numpy.all(self.y == a.y)        
    
    def testfromtable(self):
        """class method for construction from table"""
        a = segment.DataContainer.fromtable(numpy.vstack((self.x, self.y)).T)
        assert numpy.all(self.x == a.x)
        assert numpy.all(self.y == a.y)

    def testfromfile(self):
        """class method to read data from file"""
        a = segment.DataContainer.fromfile('testdata/almostlinear.dat')
    
    def testsplit(self):
        """data can be split at a position"""
        d = segment.DataContainer.fromfile('testdata/almostlinear.dat')
        a, b = d.split(2)

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.data = segment.DataContainer.fromfile('testdata/almostlinear.dat')

    def testInit(self):
        "Create fit object"
        self.fitter = segment.LinearRegression(self.data)
    
    def testFit(self):
        "Realistic linear fit"
        self.testInit()
        assert len(self.fitter.coeff) == 2, "Linear fit has two coefficients"
        assert self.fitter.error < 0.05, "Fit error is too large"

class TestTopDown(unittest.TestCase):
    def setUp(self):
        self.data = segment.DataContainer.fromfile('testdata/almostlinear.dat')
    
    def testTopDown(self):
        a = segment.topdown(self.data, 0.01, segment.LinearRegression)
        
class TestAlmostLinear(unittest.TestCase):
    def setUp(self):
        self.d = numpy.loadtxt('testdata/almostlinear.dat', skiprows=1)
        


if __name__ == '__main__':
    unittest.main() 