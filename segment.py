#!/usr/bin/env python

# Time-series segmentation module

import pylab
import numpy
import scikits.timeseries as ts
import scikits.timeseries.lib.plotlib as tsp
import scikits.timeseries.lib.moving_funcs as tsm

class DataContainer:
    """Generic container for timeseries data"""

    @staticmethod
    def fromtable(table):
        return DataContainer(table[:,0], table[:,1])

    @staticmethod
    def fromfile(filename):
        return DataContainer.fromtable(numpy.loadtxt(filename, skiprows=1))
    
    def __init__(self, x, y): 
        assert len(x) == len(y)
        self.x = x
        self.y = y
    
    def __repr__(self):
        return "DataContainer x=" + str(self.x) + ", y=" + str(self.y)
    
    def __len__(self):
        return len(self.x)
    
    def values(self):
        return self.x, self.y

    def split(self, i):
        return DataContainer(self.x[:i], self.y[:i]), DataContainer(self.x[i-1:], self.y[i-1:])

class LinearRegression:
    """ Linear regression class fits its data with straight line """
    minlength = 2
    def __init__(self, data):
        self.data = data
        self.coeff, residuals, _, _, _ = numpy.polyfit(self.data.x, self.data.y, 1, full=True)
        self.error = numpy.linalg.norm(residuals)
        self.xrange = [min(self.data.x), max(self.data.x)]
        self.liney = numpy.polyval(self.coeff, self.xrange)
    def __repr__(self):
        return "Linear (mx+c) fit through %i data points: y=%f*x+%f, error=%f" %  \
               (len(self.data), self.coeff[0], self.coeff[1], self.error)


def topdown(data, epsilon, fitter):
    """
    Top-Down fitting piecewise fitting.  fitter is a function or class
    that returns a fitted function
    """
    
    def localtopdown(d):
        #print "Calling topdown on " + str(d)
        return topdown(d, epsilon, fitter)
    fit = fitter(data)
    print data, fit.error
    if fit.error < epsilon or len(data) <= fitter.minlength:
        return [fit]
    else:
        N = len(data)
        bestsofar = numpy.inf
        bestfit = [fit]
        for i in xrange(fitter.minlength, N-fitter.minlength+1):
            print "Subdividing at position ", i
            # Do fits with subdivision at point i
            l, r = map(localtopdown, data.split(i))
            fits = l + r
            totalerror = numpy.sqrt(sum(f.error**2 for f in fits))
            # Remember best subdivision
            if totalerror < bestsofar:
                bestsofar = totalerror
                # TODO: overload operator+ for fits
                bestfit = fits
        # return best subdivision
        return bestfit


def bottomup(T, epsilon):
    pass


def testts():
    d = ts.tsfromtxt('testdata/weight.dat',
                     dateconverter=lambda s: ts.Date('D', string=s),
                     datecols=0,
                     names=['weight', 'fat'])

    weight = d['weight']
    tsp.tsplot(weight, '.')
    weight_avg = tsm.cmov_average(weight, 20)
    tsp.tsplot(weight_avg+1, weight_avg-1)
    pylab.show()


if __name__ == "__main__":
    lineartest = DataContainer.fromfile('testdata/weightindexed.dat')
    print lineartest
    fits = topdown(lineartest, 800, LinearRegression)
    for f in fits:
        pylab.plot(f.data.x, f.data.y, '.')
        pylab.plot(f.xrange, f.liney, '-')
    pylab.show()
    