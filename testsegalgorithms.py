#!/usr/bin/env python

import segment

d = segment.DataContainer.fromfile('testdata/twolinear.dat')

d.plot()
segmenter = segment.BottomUp(segment.LineThroughEndPoints, 2)

fits = segmenter.segment(d)
fits.plot()
segment.plt.show()
