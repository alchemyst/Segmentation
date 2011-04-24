#!/usr/bin/env python

import unittest
import segment
import numpy

class TestDataContainer(unittest.TestCase):
    def setUp(self):
        self.x = numpy.array([1, 2, 3])
        self.y = numpy.array([4, 5, 6])
        self.x2 = numpy.array([4, 5, 6])
        self.y2 = numpy.array([6, 5, 4])        

    def testinit(self):
        """construction from x and y data"""
        a = segment.DataContainer(self.x, self.y)
        assert numpy.all(self.x == a.x)
        assert numpy.all(self.y == a.y)        
    
    def testxrange(self):
        """ xrange property """
        a = segment.DataContainer(self.x, self.y)
        assert a.xrange == (1, 3)
    
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
        
    def testadd(self):
        """the + operator works as expected"""
        a = segment.DataContainer(self.x, self.y)
        b = segment.DataContainer(self.x2, self.y2)
        c = a + b
        assert numpy.all(c.x == numpy.append(self.x, self.x2))
        assert numpy.all(c.y == numpy.append(self.y, self.y2))
        
    def testiadd(self):
        a = segment.DataContainer(self.x, self.y)
        b = segment.DataContainer(self.x2, self.y2)
        a += b
        assert numpy.all(a.x == numpy.append(self.x, self.x2))
        assert numpy.all(a.y == numpy.append(self.y, self.y2))

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

class TestBottomUp(unittest.TestCase):
    def setUp(self):
        self.data = segment.DataContainer.fromfile('testdata/almostlinear.dat')
    
    def testBottomUp(self):
        a = segment.bottomup(self.data, 2, segment.LinearRegression, 0.2)

        
class TestAlmostLinear(unittest.TestCase):
    def setUp(self):
        self.d = numpy.loadtxt('testdata/almostlinear.dat', skiprows=1)
        


if __name__ == '__main__':
    unittest.main() 