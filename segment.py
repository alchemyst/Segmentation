#!/usr/bin/env python

from __future__ import print_function

# Time-series segmentation module

import matplotlib.pyplot as plt
import numpy
import scipy.optimize
import sys
import logging
logging.basicConfig(level=logging.WARN)


class DataContainer(object):
    """Generic container for timeseries data"""

    @staticmethod
    def fromtable(table):
        return DataContainer(table[:, 0], table[:, 1])

    @staticmethod
    def fromfile(filename):
        return DataContainer.fromtable(numpy.loadtxt(filename, skiprows=1))

    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x = numpy.asarray(x)
        self.y = numpy.asarray(y)

    @property
    def xrange(self):
        return (min(self.x), max(self.x))

    def __repr__(self):
        return "DataContainer x=" + str(self.x) + ", y=" + str(self.y)

    def __len__(self):
        return len(self.x)

    def plot(self):
        plt.plot(self.x, self.y, '.')

    def values(self):
        return self.x, self.y

    def split(self, i):
        """ split the container at point i.
        Returns: two containers both containing that point """
        return DataContainer(self.x[:i], self.y[:i]), DataContainer(self.x[i-1:], self.y[i-1:])

    def __add__(self, other):
        return DataContainer(numpy.append(self.x, other.x),
                             numpy.append(self.y, other.y))

    def merge(self, other):
        # FIXME: merge is not symmetrical to split!
        self += other

    def contains(self, x):
        # TODO: Figure out how the inequalities have to work out for consistency
        minx, maxx = self.xrange
        return minx <= x <= maxx


class Fitter:
    def __init__(self, data):
        self.data = data
        self.xrange = self.data.xrange
        self.plotevaled = False

    def calcresiduals(self):
        self.residuals = self.data.y - self.eval(self.data.x)
        self.error = numpy.linalg.norm(self.residuals)/len(self.data)

    @classmethod
    def divisions(self, N, stride):
        """ return iterator for division points of N-length data """
        return range(self.minlength, N-self.minlength+1, stride)

    def plot(self):
        self.data.plot()
        x, y = self.plotvals()
        plt.plot(x, y, '-')

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
    cost = 3    # endpoints, coeffs
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
    cost = 4  # endpoints, coeffs
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
    """ Quadratic regression class fits its data with a parabola """
    minlength = 3
    cost = 5    # endpoints, coeffs
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
    cost = 2    # endpoints
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


class FitSet(object):
    def __init__(self, fits=None):
        if fits is None:
            self.fits = []
        else:
            self.fits = fits

    @property
    def error(self):
        return numpy.linalg.norm(numpy.hstack(f.residuals for f in self.fits))

    def plot(self):
        for f in self.fits:
            f.plot()

    def append(self, fit):
        self.fits.append(fit)

    def eval(self, xs):
        if not hasattr(xs, "__iter__"):
            for fit in self.fits:
                if fit.data.contains(xs):
                    return fit.eval(xs)
            else:
                raise ValueError
        else:
            return numpy.array(map(self.eval, xs))

    def __add__(self, other):
        return FitSet(self.fits + other.fits)

    def __len__(self):
        return len(self.fits)

    def __repr__(self):
        return self.fits.__repr__()

#TODO: change logic to use FitSet class


class SegmentationAlgorithm:
    pass


class TopDown(SegmentationAlgorithm):
    name = "TopDown"

    def __init__(self, fitter, fitbudget, stride=1):
        self.fitter = fitter
        self.fitbudget = fitbudget
        self.stride = stride
        self.clearstore()

    def clearstore(self):
        self.solutionstore = {}
        self.done = False

    def topdown(self, data, fitbudget, depth=1):

        def localtopdown(d, fitb):
            problemparameters = d.xrange + (fitb,)
            if problemparameters not in self.solutionstore:
                self.solutionstore[problemparameters] = self.topdown(d, fitb, depth+1)
            return self.solutionstore[problemparameters]

        fit = self.fitter(data)
        if fitbudget == 1:
            return FitSet([fit])
        else:
            N = len(data)
            bestsofar = numpy.inf
            bestfit = FitSet([fit])
            for i in self.fitter.divisions(N, self.stride):
                # Do fits with subdivision at point i
                ldata, rdata = data.split(i)
                l = localtopdown(ldata, 1)
                r = localtopdown(rdata, fitbudget - 1)
                fits = l + r
                totalerror = fits.error/len(data)
                # Remember best subdivision
                if totalerror < bestsofar:
                    bestsofar = totalerror
                    bestfit = fits
                logging.info("%10i, %10i, %10s, %10f, %10i" %
                             (depth, len(self.solutionstore), str(data.xrange), totalerror, bestsofar))
            # return best subdivision
            return bestfit

    def segment(self, data):
        self.clearstore()
        self.fits = self.topdown(data, self.fitbudget)
        return self.fits


class BottomUp(SegmentationAlgorithm):
    name = "BottomUp"

    def __init__(self, fitter, fitbudget, stride=1, epsilon=0.2):
        self.fitter = fitter
        self.fitbudget = fitbudget
        self.stride = 1
        self.epsilon = epsilon

    def bottomup(self, data, fitbudget):
        # Seed initial data
        # NOTE: this is a really slow way of doing it.  Direct indexing would be much faster

        workingdata = []
        fits = FitSet()
        rest = data
        for i in range(len(data)//self.stride - 1):
            firstgroup, rest = rest.split(self.stride+1)
            workingdata.append(firstgroup)
            fits.append(self.fitter(firstgroup))

        initialerror = fits.error

        if initialerror > self.epsilon:
            return fits

        while len(fits) > 1:
            # build pairs of fits
            pairs = [self.fitter(fits.fits[i].data + fits.fits[i+1].data) for i in range(len(fits)-1)]
            # find best break
            # FIXME: This is very slow, but it's easy to understand
            bestbreak = 0
            for i, p in enumerate(pairs):
                if p.error < pairs[bestbreak].error:
                    bestbreak = i
            if pairs[bestbreak].error > self.epsilon:
                break
            # merge best break
            fits.fits[bestbreak] = pairs[bestbreak]
            del fits.fits[bestbreak+1]

        return fits

    def segment(self, data):
        return self.bottomup(data, self.fitbudget)


def testts():
    d = ts.tsfromtxt('testdata/weight.dat',
                     dateconverter=lambda s: ts.Date('D', string=s),
                     datecols=0,
                     names=['weight', 'fat'])

    weight = d['weight']
    tsp.tsplot(weight, '.')
    weight_avg = tsm.cmov_average(weight, 20)
    tsp.tsplot(weight_avg+1, weight_avg-1)
    plt.show()


if __name__ == "__main__":
    plotfits = True
    if len(sys.argv) < 2:
        filename = 'testdata/weightindexed.dat'
        stride = 10
    else:
        filename = sys.argv[1]
        stride = 1

    #lineartest = DataContainer.fromfile('testdata/weightindexed_small.dat')
    lineartest = DataContainer.fromfile(filename)
    fronts = []
    fitrange = range(3, 4)
    fittypes = (ConstantPiecewise,
                LinearRegression,
                QuadraticRegression,
                LineThroughEndPoints,
                ExponentialRegression,
                )
    for fittype in fittypes:
        print(fittype.description)
        allfits = []
        fiterror = []
        segmenter = TopDown(fittype, 1, stride=stride)
        for i in fitrange:
            segmenter.fitbudget = i
            print("fitting %i items" % i)
            fits = segmenter.segment(lineartest)
            allfits.append(fits)
            fiterror.append(fits.error/len(lineartest))
        fronts.append(fiterror)
        if plotfits:
            for fits in allfits:
                plt.figure()
                fits.plot()
            plt.figure()
            plt.plot(fitrange, fiterror)

    plt.figure()
    for front in fronts:
        plt.plot(fitrange, front)
    plt.legend([fit.description for fit in fittypes])
    plt.show()
