#!/usr/bin/env python

# Time-series segmentation module

import pylab
import numpy
import scipy.optimize
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
        self.xrange = (min(self.x), max(self.x))
    
    def __repr__(self):
        return "DataContainer x=" + str(self.x) + ", y=" + str(self.y)
    
    def __len__(self):
        return len(self.x)
    
    def plot(self):
        pylab.plot(self.x, self.y, '.')
    
    def values(self):
        return self.x, self.y

    def split(self, i):
        return DataContainer(self.x[:i], self.y[:i]), DataContainer(self.x[i-1:], self.y[i-1:])

class Fitter:
    def __init__(self, data):
        self.data = data
        self.xrange = self.data.xrange
        self.plotevaled = False

    def calcresiduals(self):
        self.residuals = self.data.y - self.eval(self.data.x)
        self.error = numpy.linalg.norm(self.residuals)/len(self.data)
    
    def plot(self):
        self.data.plot()
        x, y = self.plotvals()
        pylab.plot(x, y, '-')

    def plotvals(self):
        if not self.plotevaled:
            self.plotx = numpy.linspace(*self.xrange)
            self.ploty = self.eval(self.plotx)
            self.plotevaled = True
        return self.plotx, self.ploty

    def __repr__(self):
        return "Fitter object"


class ConstantPiecewise(Fitter):
    """ Constant regression class fits its data with a single average"""
    minlength = 1
    cost = 3 # endpoints, coeffs
    description = "Constants"
    def __init__(self, data):
        Fitter.__init__(self, data)
        self.value = numpy.mean(self.data.y)
        self.calcresiduals()
        self.liney = self.eval(numpy.array(self.xrange))
    
    def eval(self, x):
        rval = x.copy()
        rval.fill(self.value)
        return rval
        
    def plotvals(self):
        return self.xrange, self.liney
    
    def __repr__(self):
        return "Constant through %i data points: y = %f, error = %f" %  \
               (len(self.data),) + self.coeff + (self.error,)
        
class LinearRegression(Fitter):
    """ Linear regression class fits its data with straight line """
    minlength = 2
    cost = 4 # endpoints, coeffs
    description = "Linear regression"
    def __init__(self, data):
        Fitter.__init__(self, data)
        self.coeff = numpy.polyfit(self.data.x, self.data.y, 1)
        self.calcresiduals()
        self.liney = self.eval(self.xrange)
    
    def eval(self, x):
        return numpy.polyval(self.coeff, x)
        
    def plotvals(self):
        return self.xrange, self.liney
    
    def __repr__(self):
        return "Linear fit through %i data points: y = %f*x + %f, error = %f" %  \
               (len(self.data), self.coeff[0], self.coeff[1], self.error)

class QuadraticRegression(Fitter):
    """ Linear regression class fits its data with straight line """
    minlength = 3
    cost = 5 # endpoints, coeffs
    description = "quadratic regression"
    def __init__(self, data):
        Fitter.__init__(self, data)
        self.coeff = numpy.polyfit(self.data.x, self.data.y, 2)
        self.calcresiduals()
        self.liney = self.eval(self.xrange)
    
    def eval(self, x):
        return numpy.polyval(self.coeff, x)
        
    def __repr__(self):
        return "Quadratic fit through %i data points: y = %f*x^2 + %f*x + %f, error = %f" %  \
               (len(self.data),) + self.coeff + (self.error,)


class LineThroughEndPoints(Fitter):
    """ Straight line through data endpoints """
    minlength = 2
    cost = 2 # endpoints
    description = "Straight line through end points"
    def __init__(self, data):
        Fitter.__init__(self, data)
        self.liney = [data.y[0], data.y[-1]]
        self.coeff = numpy.polyfit(self.data.xrange, self.liney, 1)
        self.calcresiduals()

    def eval(self, x):
        return numpy.polyval(self.coeff, x)

    def plotvals(self):
        return self.xrange, self.liney
        
    def __repr__(self):
        return "Straight line through endpoints of %i data" % len(self.data)
        

class ExponentialRegression(Fitter):
    """ Exponential regression class - y = y0 + k(1-exp(-(x-x0)/tau)) """
    minlength = 9
    cost = 4
    description = "Exponential regression"
    def __init__(self, data):
        Fitter.__init__(self, data)
        
        def form(x, offset, k, tau):
            return offset + k*(1-numpy.exp(-(x-x[0])/tau))
        
        guess = [self.data.y[0], self.data.y[-1] - self.data.y[0], 20]
        try:
            popt, pcov = scipy.optimize.minpack.curve_fit(form, self.data.x, self.data.y, guess)
            self.offset, self.k, self.tau = popt
            self.optimal = True
        except RuntimeError:
            self.offset, self.k, self.tau = guess
            self.optimal = False
            
        self.calcresiduals()     
        
    def eval(self, x):
        return self.offset + self.k*(1 - numpy.exp(-(x - self.xrange[0])/self.tau))
        
    def __repr__(self):
        return "Exponential through %i data points: tau = %f, k = %f" % \
               (len(self.data), self.tau, self.k)


def fitseterror(fits):
    return numpy.linalg.norm(numpy.hstack(f.residuals for f in fits))

# For Dynamic Programming/Memoization of topdown results
solutionstore = {}
def topdown(data, fitbudget, fitter, depth=1):
    """
    Top-Down piecewise fitting.  fitter is a function or class
    that returns a fitted function
    """
    
    def localtopdown(d, fitb):
        problemparameters = d.xrange + (fitb,)
        if problemparameters not in solutionstore:
            solutionstore[problemparameters] = topdown(d, fitb, fitter, depth+1)
        return solutionstore[problemparameters]
    
    fit = fitter(data)
    if fitbudget == 1:
        return [fit]
    else:
        N = len(data)
        bestsofar = numpy.inf
        bestfit = [fit]
        for i in xrange(fitter.minlength, N-fitter.minlength+1):
            # Do fits with subdivision at point i
            ldata, rdata = data.split(i)
            l = localtopdown(ldata, 1)
            r = localtopdown(rdata, fitbudget - 1)
            fits = l + r
            totalerror = fitseterror(fits)/len(data)
            # Remember best subdivision
            if totalerror < bestsofar:
                bestsofar = totalerror
                # TODO: overload operator+ for fits
                bestfit = fits
            #print depth, len(solutionstore), data.xrange, totalerror, bestsofar
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
    plotfits = False
    lineartest = DataContainer.fromfile('testdata/weightindexed_small.dat')
    fronts = []
    fitrange = range(1, 7)
    fittypes = (ConstantPiecewise, LinearRegression, QuadraticRegression, LineThroughEndPoints, ExponentialRegression)
    for fittype in fittypes:
        solutionstore = {}
        print fittype.description
        allfits = []
        fiterror = []
        for i in fitrange:
            print "fitting %i items" % i
            fits = topdown(lineartest, i, fittype)
            allfits.append(fits)
            fiterror.append(fitseterror(fits)/len(lineartest))
        fronts.append(fiterror)
        if plotfits:
            for fits in allfits:
                pylab.figure()
                for f in fits:
                    f.plot()
                pylab.show()
            pylab.figure()
            pylab.plot(fitrange, fiterror)
            pylab.show()
    pylab.figure()
    for front in fronts:
        pylab.plot(fitrange, front)
    pylab.legend([fit.description for fit in fittypes])
    pylab.show()
