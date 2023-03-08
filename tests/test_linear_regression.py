import segment

data = segment.DataContainer.fromfile('testdata/almostlinear.dat')

def test_linear_regression():
    fitter = segment.LinearRegression(data)

    assert len(fitter.coeff) == 2, "Linear fit has two coefficients"
    assert fitter.error < 0.05, "Fit error is too large"


